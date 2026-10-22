"""Persistent SAM2 image predictor prompted by a Grounding DINO box."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT/'checkpoints/sam2.1_hiera_small.pt'
DEFAULT_CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'


class SAM2Segmenter:
    def __init__(self, device='cuda:0', checkpoint=DEFAULT_CHECKPOINT,
                 model_config=DEFAULT_CONFIG, predictor=None):
        self.model_config = model_config
        self.checkpoint = Path(checkpoint)
        if predictor is None:
            if not self.checkpoint.is_file():
                raise FileNotFoundError(f'SAM2 checkpoint not found: {self.checkpoint}. '
                                        'See docs/segmentation.md for setup.')
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor(build_sam2(model_config, str(self.checkpoint), device=device))
        self.predictor = predictor

    def segment(self, scene, box):
        import torch
        # Float32 works on the existing RTX 6000; no BF16 hardware assumption.
        with torch.inference_mode():
            try:
                self.predictor.set_image(np.array(scene.convert('RGB'), copy=True))
                masks, scores, _ = self.predictor.predict(
                    box=np.asarray(box, dtype=np.float32), multimask_output=False)
            finally:
                # Drop this capture's embeddings; keep model weights for reuse.
                self.predictor.reset_predictor()
        masks, scores = np.asarray(masks), np.asarray(scores)
        if (masks.shape != (1, scene.height, scene.width) or scores.shape != (1,)
                or not np.isfinite(masks).all() or not np.isfinite(scores).all()):
            raise ValueError('SAM2 returned invalid masks or scores')
        mask = masks[0] > 0
        if np.count_nonzero(mask) < 16:
            raise ValueError('SAM2 produced an empty or negligible foreground')
        return mask, float(scores[0])
