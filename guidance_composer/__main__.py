import argparse
import json
import os
from pathlib import Path
import sys
import time

from .segmentation import DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from .orientation import DEFAULT_CHECKPOINT as ORIENTATION_CHECKPOINT, DEFAULT_SOURCE as ORIENTATION_SOURCE

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')


def main():
    parser = argparse.ArgumentParser(description='Compose self-contained Unity guidance JSON')
    parser.add_argument('--image', required=True, help='Test harness image; loaded into memory')
    parser.add_argument('--spec', nargs='+', required=True)
    parser.add_argument('--output', help='Optional new directory for diagnostic images/GLBs and guidance.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--grounding-model', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--reconstruction-model', default='stabilityai/TripoSR')
    parser.add_argument('--sam2-checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--sam2-config', default=DEFAULT_CONFIG)
    parser.add_argument('--orientation-checkpoint', type=Path, default=ORIENTATION_CHECKPOINT)
    parser.add_argument('--orientation-source', type=Path, default=ORIENTATION_SOURCE)
    parser.add_argument('--threshold', type=float, default=.3)
    args = parser.parse_args()
    from PIL import Image, ImageOps
    from .specification import validate
    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    specs = []
    for name in args.spec:
        spec = json.loads(Path(name).read_text())
        if 'scene_image' in spec:
            raise ValueError('CLI JSON omits scene_image; --image supplies in-memory pixels')
        spec['scene_image'] = image
        validate(spec)
        specs.append(spec)
    output = Path(args.output) if args.output else None
    if output:
        output.mkdir(parents=True, exist_ok=False)
    from .grounding import GroundingDino
    from .preprocessing import ObjectPreprocessor
    from .reconstruction import TripoSR
    from .pipeline import GuidanceComposer
    from .orientation import OrientAnythingV2
    start = time.perf_counter()
    grounder = GroundingDino(args.device, args.grounding_model, args.threshold)
    try:
        composer = GuidanceComposer(grounder, TripoSR(args.device, args.reconstruction_model),
                                    ObjectPreprocessor(args.device, checkpoint=args.sam2_checkpoint,
                                                       model_config=args.sam2_config),
                                    OrientAnythingV2(args.device, checkpoint=args.orientation_checkpoint,
                                                     source=args.orientation_source))
        init_ms = (time.perf_counter()-start)*1000
        if output:
            (output/'initialization.json').write_text(json.dumps({'model_initialization_ms': init_ms}))
        failed = False
        for index, (name, spec) in enumerate(zip(args.spec, specs)):
            try:
                directory = output/f'{index:02d}_{Path(name).stem}' if output else None
                response = composer.generate(spec, directory)
                if directory:
                    print(json.dumps({'schema_version': response['schema_version'],
                                      'anchor_count': len(response['anchors']),
                                      'embedded_glb_count': len(response['assets']),
                                      'timeline_operations': len(response['timeline']),
                                      'timings_ms': response['timings_ms']}, indent=2))
                else:
                    # One self-contained JSON response per line (NDJSON for multiple specs).
                    print(json.dumps(response, allow_nan=False))
            except Exception as exc:
                failed = True
                print(f'{name}: {exc}', file=sys.stderr)
        return 1 if failed else 0
    finally:
        grounder.close()


if __name__ == '__main__':
    sys.exit(main())
