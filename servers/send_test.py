"""Send the same image + capture metadata payload expected from Unity."""
import argparse
import base64
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--camera-spec', default='examples/apple_forward.json',
                        help='Read only capture metadata from this file; no semantic fields are transmitted')
    parser.add_argument('--include-axes', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8000/guidance')
    parser.add_argument('--output', help='Optional file to save the complete response')
    parser.add_argument('--timeout', type=float, default=180.)
    args = parser.parse_args()
    capture = json.loads(Path(args.camera_spec).read_text())
    payload = {'scene_image_base64': base64.b64encode(Path(args.image).read_bytes()).decode('ascii'),
               'camera_pose': capture['camera_pose']}
    if args.include_axes:
        payload['camera_axes_world'] = capture['camera_axes_world']
    request = Request(args.url, data=json.dumps(payload, allow_nan=False).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        print(f'HTTP {exc.code}: {exc.read().decode()}', file=sys.stderr)
        return 1
    except (URLError, TimeoutError) as exc:
        print(f'Cannot reach guidance server: {exc}', file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'schema_version': result['schema_version'],
                      'anchors': len(result['anchors']), 'assets': len(result['assets']),
                      'timeline_operations': len(result['timeline']), 'timings_ms': result.get('timings_ms')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
