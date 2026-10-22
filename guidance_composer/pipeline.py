import copy
import json
from pathlib import Path
import time
import numpy as np
from PIL import ImageDraw

from .grounding import GroundingError
from .specification import validate, reference_required
from .manifest import build_guidance
from .exporting import export_diagnostics, annotation_font


class GuidanceComposer:
    """Keep models loaded. generate returns a transportable dict; debug output is optional."""
    def __init__(self, grounder, reconstructor, preprocessor, orientation=None):
        self.grounder, self.reconstructor, self.preprocessor = grounder, reconstructor, preprocessor
        self.orientation = orientation

    def generate(self, specification, output=None):
        spec = copy.deepcopy({k: v for k, v in specification.items() if k != 'scene_image'})
        spec['scene_image'] = specification.get('scene_image')
        validate(spec)
        directory = None
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        timings = {}
        scene = spec['scene_image'].convert('RGB')
        try:
            if directory:
                scene.save(directory/'original.png')
                (directory/'specification.json').write_text(json.dumps(
                    {k: v for k, v in spec.items() if k != 'scene_image'}, indent=2))
            t = time.perf_counter()
            target = self.grounder.ground(scene, spec['target_object'])
            references = {}
            if reference_required(spec):
                description = spec['action']['qualifiers']['reference_object']
                references['reference_object'] = (target if description == spec['target_object']
                                                  else self.grounder.ground(scene, description))
            timings['grounding_ms'] = (time.perf_counter()-t)*1000
            if directory:
                (directory/'grounding.json').write_text(json.dumps({'target': target, 'references': references}, indent=2))
                annotated = scene.copy()
                draw = ImageDraw.Draw(annotated)
                for label, region in [('target', target), *references.items()]:
                    draw.rectangle(region['box'], outline='cyan' if label == 'target' else 'orange', width=max(3, scene.width//700))
                    draw.text(tuple(region['box'][:2]), f"{label}: {region['score']:.2f}",
                              fill='red', font=annotation_font(scene.width), stroke_width=1, stroke_fill='white')
                annotated.save(directory/'grounding.png')
            t = time.perf_counter()
            isolated, prepared = self.preprocessor.prepare(scene, target['box'])
            timings['preprocessing_ms'] = (time.perf_counter()-t)*1000
            if directory:
                isolated.save(directory/'isolated.png')
                prepared.save(directory/'reconstruction_input.png')
                if hasattr(self.preprocessor, 'diagnostics'):
                    (directory/'preprocessing.json').write_text(json.dumps(self.preprocessor.diagnostics, indent=2))
                if hasattr(self.preprocessor, 'mask_image'):
                    self.preprocessor.mask_image.save(directory/'segmentation_mask.png')
            mesh, model_timings = self.reconstructor.reconstruct(prepared)
            timings.update(model_timings)
            if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise ValueError('Reconstruction produced an empty or invalid mesh')
            alignment_rotation = None
            if self.orientation is not None:
                t = time.perf_counter()
                alignment_rotation, orientation_diagnostics, orientation_images = self.orientation.align(mesh, isolated, spec)
                timings['orientation_estimation_ms'] = (time.perf_counter()-t)*1000
                if directory:
                    (directory/'orientation.json').write_text(json.dumps(orientation_diagnostics, indent=2, allow_nan=False))
                    for name, image in orientation_images.items():
                        image.save(directory/name)
            t = time.perf_counter()
            result = build_guidance(spec, target, references, mesh, alignment_rotation=alignment_rotation)
            timings['composition_and_glb_ms'] = (time.perf_counter()-t)*1000
            if directory:
                t = time.perf_counter()
                export_diagnostics(result, directory)
                centered = mesh.copy()
                centered.apply_translation(-centered.bounds.mean(axis=0))
                centered.export(directory/'reconstructed.glb')
                timings['diagnostic_export_and_preview_ms'] = (time.perf_counter()-t)*1000
            timings['total_request_ms'] = (time.perf_counter()-start)*1000
            result['timings_ms'] = timings
            if directory:
                (directory/'guidance.json').write_text(json.dumps(result, indent=2, allow_nan=False))
                (directory/'timings.json').write_text(json.dumps(timings, indent=2))
            return result
        except Exception as exc:
            if directory:
                failure = {'error': str(exc), 'type': type(exc).__name__, 'timings': timings}
                if isinstance(exc, GroundingError):
                    failure['candidates'] = exc.candidates
                (directory/'failure.json').write_text(json.dumps(failure, indent=2))
            raise
