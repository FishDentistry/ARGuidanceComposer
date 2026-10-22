import math
import numpy as np
from PIL import Image


class ObjectPreprocessor:
    def __init__(self):
        import rembg
        self.session = rembg.new_session('u2net', providers=['CPUExecutionProvider'])

    def isolate(self, scene, box):
        import rembg
        x0, y0, x1, y1 = box
        pad = max(x1-x0, y1-y0) * 0.05
        bounds = (max(0, math.floor(x0-pad)), max(0, math.floor(y0-pad)),
                  min(scene.width, math.ceil(x1+pad)), min(scene.height, math.ceil(y1+pad)))
        isolated = rembg.remove(scene.crop(bounds).convert('RGB'), session=self.session).convert('RGBA')
        return isolated, bounds

    def prepare(self, scene, box):
        isolated, _ = self.isolate(scene, box)
        alpha = np.array(isolated.getchannel('A'))
        ys, xs = np.where(alpha > 32)
        if len(xs) < 16:
            raise ValueError('Background removal produced an empty or negligible foreground')
        foreground = isolated.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
        side = math.ceil(max(foreground.size) / 0.85)
        padded = Image.new('RGBA', (side, side))
        padded.paste(foreground, ((side-foreground.width)//2, (side-foreground.height)//2))
        gray = Image.new('RGBA', padded.size, (128, 128, 128, 255))
        return padded, Image.alpha_composite(gray, padded).convert('RGB')
