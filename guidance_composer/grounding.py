import base64
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


class GroundingError(ValueError):
    def __init__(self, message, candidates):
        super().__init__(message)
        self.candidates = candidates


def select_candidate(candidates, size, ambiguity_margin=0.10):
    """Suppress duplicate boxes, then reject close competing hypotheses."""
    width, height = size
    kept = []
    for candidate in sorted(candidates, key=lambda c: c['score'], reverse=True):
        box = np.asarray(candidate['box'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        box[[0, 2]] = box[[0, 2]].clip(0, width)
        box[[1, 3]] = box[[1, 3]].clip(0, height)
        if min(box[2:] - box[:2]) < 2:
            continue
        duplicate = False
        for other in kept:
            b = np.asarray(other['box'])
            intersection = np.maximum(0, np.minimum(box[2:], b[2:]) - np.maximum(box[:2], b[:2])).prod()
            union = (box[2:] - box[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
            if intersection / union > 0.5:
                duplicate = True
        if not duplicate:
            kept.append({**candidate, 'box': box.tolist()})
    if not kept:
        raise GroundingError('No target detected. Try a clearer image or description.', kept)
    if len(kept) > 1 and kept[0]['score'] - kept[1]['score'] < ambiguity_margin:
        raise GroundingError('Ambiguous grounding: multiple plausible regions. Refine the description.', kept)
    return kept[0]


class GroundingDino:
    def __init__(self, device='cuda:0', model='IDEA-Research/grounding-dino-tiny', threshold=0.3):
        self.threshold = threshold
        worker = Path(__file__).with_name('grounding_worker.py')
        self.process = subprocess.Popen([sys.executable, '-u', str(worker), device, model],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        response = self.process.stdout.readline()
        if not response or not json.loads(response).get('ready'):
            self.close()
            raise RuntimeError('Grounding worker failed to initialize; check isolated dependencies and cached weights.')

    def ground(self, image, description):
        data = io.BytesIO()
        image.convert('RGB').save(data, format='PNG')
        self.process.stdin.write(json.dumps({'image': base64.b64encode(data.getvalue()).decode(),
                                            'description': description, 'threshold': self.threshold}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Grounding worker exited unexpectedly.')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return select_candidate(result['candidates'], image.size)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()
