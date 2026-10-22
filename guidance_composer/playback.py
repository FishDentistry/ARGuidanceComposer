"""Deterministic Python reference executor for the Unity manifest; no semantics."""
import numpy as np
from .frames import finite_vector, matrix_quaternion, quaternion_matrix
from .manifest import event_start


def heading_basis(manifest):
    frame = manifest['coordinate_frame']
    return np.column_stack([frame[name+'_world'] for name in ('right', 'up', 'forward')])


def resolve_position(position, anchors_world, basis):
    anchor = finite_vector(anchors_world[position['anchor_id']], 3, 'localized anchor')
    return anchor + basis @ finite_vector(position['offset_m'], 3, 'offset_m')


def slerp(a, b, amount):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    dot = float(a@b)
    if dot < 0:
        b, dot = -b, -dot
    dot = np.clip(dot, -1, 1)
    if dot > .9995:
        value = a+(b-a)*amount
    else:
        angle = np.arccos(dot)
        value = (np.sin((1-amount)*angle)*a + np.sin(amount*angle)*b)/np.sin(angle)
    return value/np.linalg.norm(value)


def sample_track(event, name, time, anchors, basis):
    keys = [key for key in event['keyframes'] if name in key]
    time = float(np.clip(time, 0, event['duration_s']))
    left, right = keys[-2:]
    for a, b in zip(keys, keys[1:]):
        if time <= b['time_s']:
            left, right = a, b
            break
    t = (time-left['time_s'])/(right['time_s']-left['time_s'])
    if event['easing'] == 'smoothstep':
        t = t*t*(3-2*t)
    if name == 'position':
        a = resolve_position(left[name], anchors, basis)
        b = resolve_position(right[name], anchors, basis)
        return a+(b-a)*t
    return slerp(left[name], right[name], t)


def evaluate(manifest, anchors_world, time_s):
    """Resolve anchors first, then interpolate world positions and heading quats.

    anchors_world is a localization snapshot. This API has no live-camera input.
    """
    basis = heading_basis(manifest)
    anchors = {a['id']: finite_vector(anchors_world[a['id']], 3, 'localized anchor').copy()
               for a in manifest['anchors']}
    duration = manifest['playback']['duration_s']
    time = max(0., float(time_s))
    if manifest['playback']['loop']:
        time %= duration
    else:
        time = min(time, duration)
    states = {v['id']: {'visible': False, 'kind': v['kind']} for v in manifest['visual_objects']}
    for event in sorted(manifest['timeline'], key=event_start):
        start = event_start(event)
        if start > time:
            continue
        state = states[event['object_id']]
        if event['type'] == 'spawn':
            state['visible'] = event['visible']
            if 'position' in event:
                state['position_world_m'] = resolve_position(event['position'], anchors, basis)
            state['rotation_xyzw_world'] = matrix_quaternion(basis@quaternion_matrix(event['rotation_xyzw']))
        elif event['type'] == 'animate_transform':
            base = sample_track(event, 'position', time-start, anchors, basis)
            rotation = quaternion_matrix(sample_track(event, 'rotation_xyzw', time-start, anchors, basis))
            pivot = event['pivot']
            if pivot.get('mode') == 'relative_to_animated_position':
                pivot_world = base + basis@np.array(pivot['offset_m'])
            else:
                pivot_world = resolve_position(pivot, anchors, basis)
            state['position_world_m'] = pivot_world + basis@rotation@basis.T@(base-pivot_world)
            state['rotation_xyzw_world'] = matrix_quaternion(basis@rotation)
        elif event['type'] == 'visibility':
            state['visible'] = event['visible']
        # hold is an explicit no-change scheduling interval.
    for visual in manifest['visual_objects']:
        if visual['kind'] == 'arrow':
            states[visual['id']]['path_world_m'] = np.array([
                resolve_position(p, anchors, basis) for p in visual['path']])
    return states
