"""Fetch pinned official Orient Anything V2 inference source and weights."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import urllib.request

from guidance_composer.orientation import (DEFAULT_SOURCE, DEFAULT_CHECKPOINT,
                                           SOURCE_REVISION, CHECKPOINT_REVISION)

CHECKPOINT_BYTES = 5_048_116_892
CHECKPOINT_SHA256 = '7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb'


def verified_checkpoint(path):
    if not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == CHECKPOINT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/SpatialVision/Orient-Anything-V2.git',
                        str(args.source)], check=True)
        subprocess.run(['git', '-C', str(args.source), 'checkout', '--detach', SOURCE_REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCE_REVISION:
        raise RuntimeError(f'Orientation source must be at {SOURCE_REVISION}; found {actual}')
    if not verified_checkpoint(args.checkpoint):
        if args.checkpoint.exists():
            raise RuntimeError(f'Existing checkpoint failed verification: {args.checkpoint}')
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.checkpoint.with_suffix('.pt.part')
        url = (f'https://huggingface.co/Viglong/OriAnyV2_ckpt/resolve/{CHECKPOINT_REVISION}'
               '/demo_ckpts/rotmod_realrotaug_best.pt')
        print('Downloading official Orient Anything V2 weights (5.05 GB)...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            while chunk := response.read(8*1024*1024):
                stream.write(chunk)
        if not verified_checkpoint(temporary):
            raise RuntimeError(f'Download failed checksum verification: {temporary}')
        temporary.replace(args.checkpoint)
    print(f'Verified source: {args.source}\nVerified checkpoint: {args.checkpoint}', flush=True)


if __name__ == '__main__':
    main()
