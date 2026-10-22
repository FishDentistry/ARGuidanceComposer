import math
import numpy as np
from PIL import Image

from .segmentation import SAM2Segmenter, DEFAULT_CHECKPOINT, DEFAULT_CONFIG


class ObjectPreprocessor:
    def __init__(self, device='cuda:0', checkpoint=DEFAULT_CHECKPOINT,
                 model_config=DEFAULT_CONFIG, segmenter=None):
        self.segmenter = segmenter if segmenter is not None else SAM2Segmenter(
            device=device, checkpoint=checkpoint, model_config=model_config)

    def isolate(self, scene, box):
        x0, y0, x1, y1 = box
        if (not np.isfinite(box).all() or x1 <= x0 or y1 <= y0
                or x0 < 0 or y0 < 0 or x1 > scene.width or y1 > scene.height):
            raise ValueError('Segmentation requires a nonempty grounded box within the scene')
        mask, score = self.segmenter.segment(scene, box)
        pad = max(x1-x0, y1-y0) * 0.05
        bounds = (max(0, math.floor(x0-pad)), max(0, math.floor(y0-pad)),
                  min(scene.width, math.ceil(x1+pad)), min(scene.height, math.ceil(y1+pad)))
        self.mask_image = Image.fromarray(mask.astype(np.uint8)*255)
        rgba = scene.convert('RGBA')
        rgba.putalpha(self.mask_image)
        self.diagnostics = {
            'strategy': 'sam2_box_prompt',
            'model_config': getattr(self.segmenter, 'model_config', None),
            'predicted_iou': score,
            'box_prompt_scene_px': list(box),
            'crop_bounds_px': list(bounds),
            'foreground_pixels': int(np.count_nonzero(mask)),
        }
        return rgba.crop(bounds), bounds

    def prepare(self, scene, box):
        isolated, _ = self.isolate(scene, box)
        alpha = np.array(isolated.getchannel('A'))
        ys, xs = np.where(alpha > 32)
        if len(xs) < 16:
            raise ValueError('Segmentation produced an empty or negligible foreground in the grounded crop')
        foreground = isolated.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
        side = math.ceil(max(foreground.size) / 0.85)
        padded = Image.new('RGBA', (side, side))
        padded.paste(foreground, ((side-foreground.width)//2, (side-foreground.height)//2))
        gray = Image.new('RGBA', padded.size, (128, 128, 128, 255))
        return padded, Image.alpha_composite(gray, padded).convert('RGB')
