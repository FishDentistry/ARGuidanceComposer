"""Orient Anything V2 relative pose, converted to the existing Unity alignment."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .frames import axis_quaternion, captured_frame, matrix_quaternion, quaternion_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT/'third_party/orient_anything_v2'
DEFAULT_CHECKPOINT = ROOT/'checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt'
SOURCE_REVISION = '73b11c9dc83e84daeb563d0c766831f2c66b0a18'
CHECKPOINT_REVISION = '34fe2ecbdb987a0fb98437af00600f22a0fdfb99'
# Upstream relate_azi_ele_rot_to_OneposeRMatrix converts its X-view/Z-up
# coordinate system into OpenCV's right/down/forward camera coordinates.
OA_TO_CV = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
CV_TO_UNITY_CAMERA = np.diag([1., -1., 1.])
GLTF_TO_UNITY = np.diag([1., 1., -1.])


def rotation_matrix(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (3, 3) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(result), 1., atol=1e-5)):
        raise ValueError('Orientation requires a finite proper rotation matrix')
    return result


def relative_angles_to_cv(angles):
    azimuth, elevation, roll = np.asarray(angles, dtype=float)
    if not np.isfinite([azimuth, elevation, roll]).all():
        raise ValueError('Orient Anything returned nonfinite angles')
    # Exactly the order/sign of upstream azi_ele_rot_to_Obj_Rmatrix_batch.
    relative = (quaternion_matrix(axis_quaternion([1, 0, 0], roll))
                @ quaternion_matrix(axis_quaternion([0, 1, 0], elevation))
                @ quaternion_matrix(axis_quaternion([0, 0, 1], -azimuth)))
    return OA_TO_CV @ relative @ OA_TO_CV.T


def mesh_alignment_rotation(relative_cv, reference_camera_from_mesh, spec):
    frame = captured_frame(spec['camera_pose'], spec['camera_axes_world'])
    camera_world = quaternion_matrix(spec['camera_pose']['rotation'])
    # The GLB importer has already reflected Z. The two reflections here
    # cancel handedness, leaving a proper quaternion rotation for alignment.
    alignment = (frame.basis.T @ camera_world @ CV_TO_UNITY_CAMERA
                 @ rotation_matrix(relative_cv) @ rotation_matrix(reference_camera_from_mesh)
                 @ GLTF_TO_UNITY)
    return matrix_quaternion(rotation_matrix(alignment))


def white_square(image, side=518):
    image = image.convert('RGBA')
    bounds = image.getchannel('A').getbbox()
    if bounds is None:
        raise ValueError('Orientation input has no foreground')
    image = image.crop(bounds)
    extent = int(np.ceil(max(image.size)/.85))
    square = Image.new('RGBA', (extent,)*2, 'white')
    square.alpha_composite(image, ((square.width-image.width)//2, (square.height-image.height)//2))
    return square.convert('RGB').resize((side, side), Image.Resampling.BICUBIC)


def reference_camera(mesh):
    # Choose an informative broadside view, rather than looking down a can's
    # length or at the edge of a flat object. PCA chooses only the render camera;
    # it does not establish physical up. The learned relative pose determines
    # the final alignment, which can point in any world direction.
    _, axes = np.linalg.eigh(np.cov(np.asarray(mesh.vertices).T))
    forward = axes[:, 0]
    down = -axes[:, 2]
    right = np.cross(down, forward)
    return np.row_stack([right, down, forward])


def render_mesh(mesh, camera_from_mesh=None, side=518):
    """Headless EGL render, with an explicit OpenCV camera rotation.

    Rendering normalizes only position/size; the supplied mesh axes are retained.
    Colors come from the reconstructed asset, before ghost material replacement.
    """
    import moderngl
    rotation = rotation_matrix(reference_camera(mesh) if camera_from_mesh is None else camera_from_mesh)
    radius = np.linalg.norm(mesh.extents)/2
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Cannot render an empty mesh for orientation')
    vertices = (mesh.vertices-mesh.bounds.mean(axis=0))/radius
    visual = mesh.visual.to_color() if mesh.visual.kind == 'texture' else mesh.visual
    colors = np.asarray(visual.vertex_colors)[:, :3]/255.
    data = np.column_stack([vertices, mesh.vertex_normals, colors]).astype('f4')
    context = moderngl.create_standalone_context(backend='egl')
    try:
        program = context.program(vertex_shader='''
            #version 330
            in vec3 position;
            in vec3 normal;
            in vec3 color;
            uniform mat3 camera;
            out vec3 vertex_color;
            void main() {
                vec3 p = camera * position + vec3(0, 0, 3.0);
                float f = 2.41421356;
                gl_Position = vec4(f*p.x, -f*p.y, 1.002002*p.z-0.2002002, p.z);
                vec3 n = normalize(camera * normal);
                float light = 0.5 + 0.5*max(0.0, dot(n, normalize(vec3(-0.4,-0.6,-1))));
                vertex_color = color*light;
            }
        ''', fragment_shader='''
            #version 330
            in vec3 vertex_color;
            out vec4 pixel;
            void main() { pixel = vec4(vertex_color, 1.0); }
        ''')
        program['camera'].write(rotation.T.astype('f4').tobytes())
        vertex_buffer = context.buffer(data.tobytes())
        indices = context.buffer(np.asarray(mesh.faces, dtype='i4').tobytes())
        array = context.vertex_array(program, [(vertex_buffer, '3f 3f 3f', 'position', 'normal', 'color')], indices)
        framebuffer = context.simple_framebuffer((side, side), components=4)
        framebuffer.use()
        context.enable(moderngl.DEPTH_TEST)
        framebuffer.clear(1., 1., 1., 0.)
        array.render()
        image = Image.frombytes('RGBA', (side, side), framebuffer.read(components=4, alignment=1))
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), rotation
    finally:
        context.release()


class OrientAnythingV2:
    def __init__(self, device='cuda:0', checkpoint=DEFAULT_CHECKPOINT, source=DEFAULT_SOURCE, model=None):
        import torch
        self.device = torch.device(device)
        if model is None:
            checkpoint, source = Path(checkpoint), Path(source)
            if not checkpoint.is_file() or not (source/'vision_tower.py').is_file():
                raise FileNotFoundError('Orient Anything V2 source/weights missing. Run '
                                        '.venv310/bin/python -m servers.setup_orientation; see docs/orientation.md.')
            sys.path.insert(0, str(source.resolve()))
            from vision_tower import VGGT_OriAny_Ref
            dtype = (torch.bfloat16 if self.device.type == 'cuda'
                     and torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16)
            model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            model.load_state_dict(state, assign=True)
            model.eval().to(self.device)
        self.model = model

    def predict_pair(self, reference, target):
        import torch
        images = [white_square(image) for image in (reference, target)]
        batch = torch.stack([torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()/255.
                             for image in images]).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch).reshape(2, 900).float()
        if not torch.isfinite(logits).all():
            raise ValueError('Orient Anything V2 produced nonfinite predictions')
        angles = torch.stack([logits[:, :360].argmax(-1), logits[:, 360:540].argmax(-1)-90,
                              logits[:, 540:900].argmax(-1)-180], dim=-1).cpu().tolist()
        return {'reference_angles_deg': angles[0], 'relative_angles_deg': angles[1]}

    def align(self, mesh, isolated, spec):
        reference, render_rotation = render_mesh(mesh)
        prediction = self.predict_pair(reference, isolated)
        relative = relative_angles_to_cv(prediction['relative_angles_deg'])
        rotation = mesh_alignment_rotation(relative, render_rotation, spec)
        aligned_render, _ = render_mesh(mesh, relative @ render_rotation)
        diagnostics = {
            'method': 'orient_anything_v2_relative_render_to_capture',
            'source_revision': SOURCE_REVISION, 'checkpoint_revision': CHECKPOINT_REVISION,
            **prediction,
            'reference_camera_from_mesh_cv': render_rotation.tolist(),
            'relative_rotation_cv': relative.tolist(),
            'mesh_alignment_rotation_xyzw': rotation,
            'orientation_status': 'estimated_orient_anything_v2',
            'symmetric_objects_may_have_ambiguous_yaw': True,
        }
        images = {'orientation_reference.png': white_square(reference),
                  'orientation_target.png': white_square(isolated),
                  'orientation_aligned.png': white_square(aligned_render)}
        return rotation, diagnostics, images
