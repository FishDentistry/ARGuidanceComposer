"""Testing-only HTTP bridge: Unity capture -> example semantics -> Unity manifest."""
import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import logging
import math
import os
from pathlib import Path
import warnings
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from guidance_composer.frames import axes_from_pose
from guidance_composer.grounding import GroundingError
from guidance_composer.segmentation import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from guidance_composer.orientation import DEFAULT_CHECKPOINT as ORIENTATION_CHECKPOINT, DEFAULT_SOURCE as ORIENTATION_SOURCE
from guidance_composer.specification import validate

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 24*1024*1024
MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger(__name__)


class CameraPose(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, frozen=True)
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]

    @field_validator('rotation')
    @classmethod
    def unit_quaternion(cls, value):
        if not math.isclose(math.sqrt(sum(x*x for x in value)), 1., abs_tol=1e-5):
            raise ValueError('rotation must be a unit quaternion [x,y,z,w]')
        return value


class CameraAxes(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, frozen=True)
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]


class SceneRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    scene_image_base64: str = Field(min_length=1, max_length=4*((MAX_IMAGE_BYTES+2)//3),
                                   description='Raw Base64 PNG/JPEG bytes (first frame for MPO), without a data-URL prefix')
    camera_pose: CameraPose
    camera_axes_world: CameraAxes | None = Field(
        default=None, description='Optional reliable captured heading; otherwise derived from the pose')


@dataclass(frozen=True)
class Settings:
    example_name: str = 'apple_forward.json'
    device: str = 'cuda:0'
    examples_dir: Path = ROOT/'examples'
    outputs_dir: Path = ROOT/'outputs'
    sam2_checkpoint: Path = DEFAULT_CHECKPOINT
    sam2_config: str = DEFAULT_CONFIG
    orientation_checkpoint: Path = ORIENTATION_CHECKPOINT
    orientation_source: Path = ORIENTATION_SOURCE

    @classmethod
    def from_env(cls):
        return cls(example_name=os.environ.get('GUIDANCE_EXAMPLE', 'pop_can_test.json'),
                   device=os.environ.get('GUIDANCE_DEVICE', 'cuda:0'),
                   sam2_checkpoint=Path(os.environ.get('GUIDANCE_SAM2_CHECKPOINT', DEFAULT_CHECKPOINT)),
                   sam2_config=os.environ.get('GUIDANCE_SAM2_CONFIG', DEFAULT_CONFIG),
                   orientation_checkpoint=Path(os.environ.get('GUIDANCE_ORIENTATION_CHECKPOINT', ORIENTATION_CHECKPOINT)),
                   orientation_source=Path(os.environ.get('GUIDANCE_ORIENTATION_SOURCE', ORIENTATION_SOURCE)))

    @property
    def example_path(self):
        # Configuration selects a filename, never a client-supplied filesystem path.
        if Path(self.example_name).name != self.example_name or not self.example_name.endswith('.json'):
            raise ValueError('GUIDANCE_EXAMPLE must be a JSON filename within examples/')
        return self.examples_dir/self.example_name


class ExampleError(RuntimeError):
    pass


def load_template(settings):
    try:
        value = json.loads(settings.example_path.read_text())
        if not isinstance(value, dict) or not all(k in value for k in ('target_object', 'action')):
            raise ValueError('example must contain target_object and action')
        return value
    except (OSError, ValueError) as exc:
        raise ExampleError(f'Cannot load configured example {settings.example_name}: {exc}') from exc


def decode_image(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError('scene_image_base64 must contain raw Base64 image bytes') from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('scene image exceeds the 24 MiB encoded-file limit')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in ('PNG', 'JPEG', 'MPO'):
                    raise ValueError('scene image must be PNG or JPEG')
                if source.width*source.height > MAX_IMAGE_PIXELS:
                    raise ValueError('scene image exceeds the 40 megapixel limit')
                return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('scene image is invalid or cannot be decoded') from exc


def create_composer(settings):
    """Called once, in the same dedicated thread used for every inference."""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from guidance_composer.grounding import GroundingDino
    from guidance_composer.pipeline import GuidanceComposer
    from guidance_composer.preprocessing import ObjectPreprocessor
    from guidance_composer.reconstruction import TripoSR
    from guidance_composer.orientation import OrientAnythingV2
    grounder = GroundingDino(settings.device)
    try:
        return GuidanceComposer(grounder, TripoSR(settings.device),
                                ObjectPreprocessor(settings.device, checkpoint=settings.sam2_checkpoint,
                                                   model_config=settings.sam2_config),
                                OrientAnythingV2(settings.device, checkpoint=settings.orientation_checkpoint,
                                                 source=settings.orientation_source))
    except BaseException:
        grounder.close()
        raise


def close_composer(composer):
    close = getattr(composer.grounder, 'close', None)
    if close is not None:
        close()


def compose_capture(settings, composer, message):
    spec = load_template(settings)  # Reload semantic fields so edits apply to the next request.
    spec['scene_image'] = decode_image(message.scene_image_base64)
    spec['camera_pose'] = message.camera_pose.model_dump(mode='json')
    # Never reuse example camera metadata for a new Unity capture.
    spec['camera_axes_world'] = (message.camera_axes_world.model_dump(mode='json')
                                 if message.camera_axes_world else axes_from_pose(spec['camera_pose']))
    validate(spec)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = settings.outputs_dir/f'server_{Path(settings.example_name).stem}_{stamp}_{uuid4().hex[:8]}'
    logger.info('Saving guidance request artifacts to %s', output)
    return composer.generate(spec, output=output)


def create_app(settings=None, composer_factory=None):
    settings = settings or Settings.from_env()
    factory = composer_factory or create_composer

    @asynccontextmanager
    async def lifespan(app):
        load_template(settings)  # Fail before loading models if configuration is invalid.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='guidance-model')
        loop = asyncio.get_running_loop()
        composer = None
        app.state.ready = False
        try:
            composer = await loop.run_in_executor(executor, factory, settings)
            app.state.composer = composer
            app.state.executor = executor
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                if composer is not None:
                    # This runs after queued inference, on the same model thread.
                    await loop.run_in_executor(executor, close_composer, composer)
            finally:
                await asyncio.to_thread(executor.shutdown, wait=True)
                app.state.composer = None

    app = FastAPI(title='Guidance Composer Testing Server', version='1.0', lifespan=lifespan)
    app.state.ready = False

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Avoid echoing a large Base64 payload (or nonfinite numbers) in error responses.
        return JSONResponse(status_code=422, content={'detail': {
            'code': 'invalid_request',
            'errors': [{'loc': list(e['loc']), 'message': e['msg'], 'type': e['type']} for e in exc.errors()]}})

    @app.get('/health')
    async def health():
        if not app.state.ready:
            raise HTTPException(503, detail={'code': 'not_ready'})
        return {'status': 'ready', 'example': settings.example_name,
                'device': settings.device, 'requests_serialized': True}

    @app.post('/guidance', response_model=None, summary='Generate a complete Unity guidance manifest')
    async def guidance(message: SceneRequest):
        if not app.state.ready:
            raise HTTPException(503, detail={'code': 'not_ready'})
        try:
            # A single-worker executor prevents overlapping model use even if a
            # disconnected HTTP request is cancelled while its inference continues.
            return await asyncio.get_running_loop().run_in_executor(
                app.state.executor, compose_capture, settings, app.state.composer, message)
        except GroundingError as exc:
            ambiguous = len(exc.candidates) > 1
            raise HTTPException(409 if ambiguous else 422, detail={
                'code': 'ambiguous_grounding' if ambiguous else 'grounding_failed',
                'message': str(exc), 'candidates': exc.candidates}) from exc
        except ExampleError as exc:
            logger.exception('Configured guidance example is unavailable')
            raise HTTPException(503, detail={'code': 'example_unavailable',
                                             'message': 'The configured example cannot be loaded; check server logs.'}) from exc
        except ValueError as exc:
            raise HTTPException(422, detail={'code': 'invalid_capture_or_spec', 'message': str(exc)}) from exc
        except Exception as exc:
            logger.exception('Guidance composition failed')
            raise HTTPException(500, detail={'code': 'generation_failed',
                                             'message': 'Guidance generation failed; check server logs.'}) from exc

    return app


app = create_app()
