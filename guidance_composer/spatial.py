"""Anchor-relative metric positions, independent of mesh shape or scale."""
import math
import numpy as np
from .frames import axis_quaternion

IDENTITY = [0., 0., 0., 1.]
OFFSETS = {'right': [1, 0, 0], 'right_of': [1, 0, 0],
           'left': [-1, 0, 0], 'left_of': [-1, 0, 0],
           'up': [0, 1, 0], 'down': [0, -1, 0],
           'forward': [0, 0, 1], 'behind': [0, 0, 1],
           'backward': [0, 0, -1], 'in_front_of': [0, 0, -1]}


def anchor_position(anchor_id, offset=(0, 0, 0)):
    return {'anchor_id': anchor_id, 'offset_m': list(map(float, offset))}


def image_anchor(anchor_id, description, region, image_size):
    width, height = image_size
    x0, y0, x1, y1 = map(float, region['box'])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('grounded anchor box is outside the source image')
    return {'id': anchor_id, 'description': description,
            'image_uv': [(x0+x1)/(2*width), (y0+y1)/(2*height)],
            'point_semantics': 'visible_grounded_bbox_center',
            'source_image_bbox_uv': [x0/width, y0/height, x1/width, y1/height]}


def motion_plan(spec, target_id='anchor_target', reference_id='anchor_reference', duration=1.5):
    q = spec['action']['qualifiers']
    start = anchor_position(target_id)
    end = anchor_position(target_id)
    rotation = q if spec['action']['type'] == 'rotate' else q.get('rotation')
    if spec['action']['type'] == 'move':
        offset = np.array(OFFSETS[q.get('placement', q.get('direction'))])*float(q['distance'])
        end = anchor_position(reference_id if 'placement' in q else target_id, offset)
    angle = 0.
    if rotation is not None:
        angle = float(rotation['angle'])
        # Unity AngleAxis(+Y,+90) turns +Z toward +X: clockwise from above.
        if rotation.get('direction') == 'counterclockwise':
            angle *= -1
        if rotation.get('viewing_convention') == 'from_negative_axis':
            angle *= -1
    count = max(1, math.ceil(abs(angle)/90))
    axis = [0, 1, 0] if rotation is None else np.eye(3)['xyz'.index(rotation['axis'])]
    keys = []
    for i in range(count+1):
        key = {'time_s': duration*i/count, 'rotation_xyzw': axis_quaternion(axis, angle*i/count)}
        if i == 0:
            key['position'] = start
        if i == count:
            key['position'] = end
        keys.append(key)
    return {'start': start, 'end': end, 'keyframes': keys, 'duration_s': duration,
            'pivot': {'mode': 'relative_to_animated_position',
                      'offset_m': list(map(float, (rotation or {}).get('pivot_offset_m', [0, 0, 0])))},
            'signed_angle_degrees': angle}
