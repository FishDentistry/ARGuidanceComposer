"""Unity +Z-forward captured-heading frame and quaternion helpers."""
from dataclasses import dataclass
import numpy as np
import trimesh


class HeadingError(ValueError):
    pass


def finite_vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {length} finite numbers')
    return result


def quaternion_matrix(q):
    q = finite_vector(q, 4, 'quaternion [x,y,z,w]')
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
        raise ValueError('quaternion must have unit length')
    return trimesh.transformations.quaternion_matrix(q[[3, 0, 1, 2]])[:3, :3]


def matrix_quaternion(rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    return trimesh.transformations.quaternion_from_matrix(matrix)[[1, 2, 3, 0]].tolist()


def axis_quaternion(axis, angle_degrees):
    axis = finite_vector(axis, 3, 'rotation axis')
    if np.linalg.norm(axis) < 1e-8:
        raise ValueError('rotation axis is zero')
    axis /= np.linalg.norm(axis)
    half = np.deg2rad(angle_degrees)/2
    return [*(axis*np.sin(half)), float(np.cos(half))]


@dataclass(frozen=True)
class CapturedFrame:
    right: tuple
    up: tuple
    forward: tuple
    pose_consistency: str = 'verified'

    @property
    def basis(self):
        return np.column_stack([self.right, self.up, self.forward])

    def as_dict(self):
        return {'type': 'camera_heading_at_capture', 'units': 'meters',
                'handedness': 'Unity_left_handed_y_up_z_forward',
                'right_world': list(self.right), 'up_world': list(self.up),
                'forward_world': list(self.forward),
                'rotation_xyzw': matrix_quaternion(self.basis),
                'frozen_at_capture': True, 'pose_consistency': self.pose_consistency}


def axes_from_pose(camera_pose):
    """For capture producers/examples. Supplied axes are required by the parser."""
    forward = quaternion_matrix(camera_pose['rotation']) @ [0., 0., 1.]
    forward[1] = 0
    if np.linalg.norm(forward) < .001:
        raise HeadingError('Nearly vertical camera: provide a reliable captured heading in camera_axes_world.')
    forward /= np.linalg.norm(forward)
    up = np.array([0., 1., 0.])
    right = np.cross(up, forward)
    return {'right': right.tolist(), 'up': up.tolist(), 'forward': forward.tolist()}


def captured_frame(camera_pose, camera_axes_world):
    finite_vector(camera_pose['position'], 3, 'camera position')
    rotation = quaternion_matrix(camera_pose['rotation'])
    if not isinstance(camera_axes_world, dict):
        raise ValueError('camera_axes_world with right/up/forward is required')
    right, up, forward = [finite_vector(camera_axes_world[k], 3, f'camera {k}')
                          for k in ('right', 'up', 'forward')]
    basis = np.column_stack([right, up, forward])
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
        raise ValueError('camera axes must be normalized and mutually orthogonal')
    if not np.allclose(up, [0, 1, 0], atol=1e-5) or abs(right[1])+abs(forward[1]) > 1e-5:
        raise ValueError('camera axes must be gravity aligned with world +Y up')
    if not np.allclose(np.cross(up, forward), right, atol=1e-5):
        raise ValueError('camera right must equal cross(world up, forward) in Unity coordinates')
    projected = rotation @ [0., 0., 1.]
    projected[1] = 0
    status = 'verified'
    if np.linalg.norm(projected) >= .001:
        projected /= np.linalg.norm(projected)
        if not np.allclose(projected, forward, atol=1e-4):
            raise ValueError('camera_axes_world is inconsistent with captured camera_pose heading')
    else:
        status = 'pose_nearly_vertical; reliable supplied captured axes used'
    # Copy to immutable tuples; later headset changes cannot affect this frame.
    return CapturedFrame(tuple(right), tuple(up), tuple(forward), status)
