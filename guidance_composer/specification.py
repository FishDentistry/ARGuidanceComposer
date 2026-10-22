"""Minimal semantic input validation; visualization decisions live in the composer."""
import math
from PIL import Image
from .frames import captured_frame, finite_vector

DIRECTIONS = {'right', 'left', 'up', 'down', 'forward', 'backward', 'behind', 'in_front_of'}
PLACEMENTS = {'right_of', 'left_of', 'in_front_of', 'behind'}


def reference_required(spec):
    return spec['action']['type'] == 'move' and 'placement' in spec['action']['qualifiers']


def validate_rotation(q):
    angle = float(q['angle'])
    if not math.isfinite(angle) or q.get('unit') != 'degrees':
        raise ValueError('rotation requires a finite angle and unit degrees')
    if q.get('axis') not in ('x', 'y', 'z'):
        raise ValueError('rotation requires an explicit x/y/z axis')
    if 'direction' in q and (q['direction'] not in ('clockwise', 'counterclockwise') or angle < 0):
        raise ValueError('rotation direction requires clockwise/counterclockwise and nonnegative angle')
    view = q.get('viewing_convention')
    if q['axis'] == 'y' and view not in (None, 'from_above', 'from_positive_axis'):
        raise ValueError('vertical rotations are viewed from above')
    if q['axis'] != 'y' and view not in ('from_positive_axis', 'from_negative_axis'):
        raise ValueError('nonvertical rotations require an explicit viewing_convention')
    if 'pivot_offset_m' in q:
        finite_vector(q['pivot_offset_m'], 3, 'pivot_offset_m')


def validate_action(spec):
    frame = captured_frame(spec['camera_pose'], spec.get('camera_axes_world'))
    action = spec['action']
    q = action['qualifiers']
    if q.get('relative_to', 'camera') != 'camera':
        raise ValueError('directions always use the frozen captured camera axes')
    if 'distance_mode' in q:
        raise ValueError('distance_mode is obsolete: placements now measure between anchors, never boundaries')
    if action['type'] == 'move':
        if ('direction' in q) == ('placement' in q):
            raise ValueError('move requires exactly one of direction or placement')
        if 'direction' in q and q['direction'] not in DIRECTIONS:
            raise ValueError('unsupported direction; use placement for left_of/right_of')
        if 'placement' in q and q['placement'] not in PLACEMENTS:
            raise ValueError('unsupported placement')
        distance = float(q['distance'])
        if not math.isfinite(distance) or distance < 0 or q.get('unit') != 'meters':
            raise ValueError('move requires a finite nonnegative distance and unit meters')
        if 'rotation' in q:
            validate_rotation(q['rotation'])
    elif action['type'] == 'rotate':
        validate_rotation(q)
    else:
        raise ValueError('supported actions are move and rotate')
    for key in ('reference_object', 'surface'):
        if key in q and (not isinstance(q[key], str) or not q[key].strip()):
            raise ValueError(f'{key} must be a nonempty natural-language description')
    if reference_required(spec) and not q.get('reference_object'):
        raise ValueError('placement requires a reference_object description')
    return frame


def validate(spec):
    if not isinstance(spec.get('scene_image'), Image.Image):
        raise ValueError('scene_image must be an in-memory PIL image')
    if not isinstance(spec.get('target_object'), str) or not spec['target_object'].strip():
        raise ValueError('target_object must be a nonempty natural-language description')
    if not isinstance(spec.get('constraints', []), list) or not all(isinstance(c, str) for c in spec.get('constraints', [])):
        raise ValueError('constraints must be a list of strings')
    if not isinstance(spec.get('context', ''), str):
        raise ValueError('context must be text')
    return validate_action(spec)
