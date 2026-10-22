"""Adapter around the existing, unmodified TripoSR Python API."""
from pathlib import Path
import sys
import time
from typing import Protocol


class Reconstructor(Protocol):
    def reconstruct(self, image):
        """Return (trimesh.Trimesh, stage_timings_ms)."""
        ...


class TripoSR:
    def __init__(self, device='cuda:0', model='stabilityai/TripoSR', resolution=64, chunk_size=4096):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'TripoSR'))
        from tsr.system import TSR
        self.device, self.resolution = device, resolution
        self.model = TSR.from_pretrained(model, config_name='config.yaml', weight_name='model.ckpt')
        self.model.renderer.set_chunk_size(chunk_size)
        self.model.eval().to(device)

    def sync(self):
        import torch
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def reconstruct(self, image):
        import torch
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            codes = self.model([image], device=self.device)
        self.sync()
        inferred = time.perf_counter()
        mesh = self.model.extract_mesh(codes, True, resolution=self.resolution)[0]
        self.sync()
        return mesh, {'reconstruction_ms': (inferred-start)*1000,
                      'mesh_extraction_ms': (time.perf_counter()-inferred)*1000}
