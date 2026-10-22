"""Self-contained Unity guidance contract, composition helpers, and validation."""
import base64
import copy
import hashlib
import json
import math
import struct
import numpy as np

from .frames import finite_vector, quaternion_matrix
from .spatial import IDENTITY, image_anchor, motion_plan
from .specification import reference_required, validate_action


def position(anchor_id, offset=(0, 0, 0)):
    return {'anchor_id': anchor_id, 'offset_m': list(map(float, offset))}


class TimelineBuilder:
    """Remote-side composition. Times are absolute; keyframe times are clip-local."""
    def __init__(self, manifest):
        self.manifest = manifest

    def spawn(self, object_id, at=0., position=None, rotation=IDENTITY):
        event = {'type': 'spawn', 'time_s': float(at), 'object_id': object_id,
                 'rotation_xyzw': list(rotation), 'visible': True}
        if position is not None:
            event['position'] = copy.deepcopy(position)
        self.manifest['timeline'].append(event)
        return self

    def animate(self, object_id, keys, at=.5, duration=1.5, easing='linear', pivot=None):
        self.manifest['timeline'].append({
            'type': 'animate_transform', 'object_id': object_id, 'start_time_s': float(at),
            'duration_s': float(duration), 'position_interpolation': 'linear',
            'rotation_interpolation': 'slerp_shortest', 'easing': easing,
            'easing_scope': 'each_keyframe_segment',
            'pivot': copy.deepcopy(pivot or {'mode': 'relative_to_animated_position', 'offset_m': [0., 0., 0.]}),
            'keyframes': copy.deepcopy(keys)})
        return self

    def hold(self, object_id, at, duration):
        self.manifest['timeline'].append({'type': 'hold', 'object_id': object_id,
                                          'start_time_s': float(at), 'duration_s': float(duration)})
        return self

    def visibility(self, object_id, at, visible):
        self.manifest['timeline'].append({'type': 'visibility', 'object_id': object_id,
                                          'time_s': float(at), 'visible': bool(visible)})
        return self

    def finish(self, loop=True, tail_hold=.25):
        self.manifest['timeline'].sort(key=event_start)
        end = max((event_end(e) for e in self.manifest['timeline']), default=0.)
        self.manifest['playback'] = {'loop': bool(loop), 'restart_on_loop': True,
                                    'duration_s': end+float(tail_hold),
                                    'anchor_resolution': 'snapshot_at_playback_start',
                                    'reset_visual_state_on_loop': True}
        validate_manifest(self.manifest)
        return self.manifest


def event_start(event):
    return event.get('start_time_s', event.get('time_s', 0.))


def event_end(event):
    return event_start(event)+event.get('duration_s', 0.)


def build_guidance(spec, target, references, mesh, alignment_rotation=None):
    """Make composition decisions remotely; return only JSON-compatible data."""
    frame = validate_action(spec)
    if alignment_rotation is not None:
        quaternion_matrix(alignment_rotation)
    target_anchor = image_anchor('anchor_target', spec['target_object'], target, spec['scene_image'].size)
    anchors = [target_anchor]
    if reference_required(spec):
        anchors.append(image_anchor('anchor_reference', spec['action']['qualifiers']['reference_object'],
                                    references['reference_object'], spec['scene_image'].size))
    extent = float(np.max(mesh.extents))
    if not math.isfinite(extent) or extent <= 0 or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError('invalid reconstructed mesh')
    # Asset retains mesh-local axes. Center only its arbitrary origin.
    centered = mesh.copy()
    center = centered.bounds.mean(axis=0)
    centered.apply_translation(-center)
    from .exporting import ghost_mesh
    blob = ghost_mesh(centered).export(file_type='glb')
    plan = motion_plan(spec)
    scale = .12/extent  # Presentation policy, NOT a physical size estimate.
    result = {
        'schema_version': '1.0',
        'coordinate_frame': frame.as_dict(),
        'capture': {'camera_pose': copy.deepcopy(spec['camera_pose']),
                    'source_image': {'width': spec['scene_image'].width, 'height': spec['scene_image'].height,
                                     'uv_origin': 'top_left', 'u_direction': 'right', 'v_direction': 'down'}},
        'execution': {'position_space': 'anchor_relative_frozen_heading_meters',
                      'rotation_space': 'frozen_heading_xyzw',
                      'world_position_rule': 'anchor_world_plus_heading_basis_times_offset_m',
                      'world_rotation_rule': 'heading_rotation_times_keyframe_rotation',
                      'resolve_keyframe_anchors_before_interpolation': True,
                      'appearance_rule': 'override_material_rgba; do_not_multiply_alpha',
                      'pivot_rule': 'root_world = pivot_world + heading * q * inverse_heading * (base_world - pivot_world)',
                      'mesh_transform_order': 'import_to_unity_then_alignment_TRS_then_animated_root'},
        'anchors': anchors,
        'assets': [{'id': 'mesh_target', 'format': 'glb', 'encoding': 'base64',
                    'data': base64.b64encode(blob).decode('ascii'), 'sha256': hashlib.sha256(blob).hexdigest(),
                    'byte_length': len(blob), 'coordinate_system': 'gltf_right_handed',
                    'unity_import': {'reflect_vertex_and_normal_z': True, 'reverse_triangle_winding': True,
                                     'apply_once': True},
                    'mesh_center_before_recentering': center.tolist()}],
        'visual_objects': [{
            'id': 'ghost_target', 'kind': 'ghost', 'asset_id': 'mesh_target',
            'mesh_alignment': {'position_m': [0., 0., 0.],
                               'rotation_xyzw': list(alignment_rotation) if alignment_rotation is not None else IDENTITY.copy(),
                               'scale': [scale]*3, 'space': 'frozen_heading_after_glb_to_unity_import',
                               'scale_status': 'illustrative_not_measured',
                               'orientation_status': ('estimated_orient_anything_v2' if alignment_rotation is not None
                                                      else 'uncalibrated'),
                               'display_max_extent_m': .12},
            'appearance': {'color_rgba': [.2, .7, 1., .4], 'opacity': .4}},
            {'id': 'highlight_target', 'kind': 'highlight', 'shape': 'wire_sphere',
             'radius_m': .075, 'appearance': {'color_rgba': [1., .75, .1, .7], 'opacity': .7}}],
        'timeline': [],
        'limitations': {'anchor_positions': 'localized accurately by Unity; not inferred remotely',
                        'metric_offsets': 'exact meters once anchors are localized',
                        'mesh_scale': 'fixed display extent; not a measured object size',
                        'mesh_orientation': ('estimated from mesh render and capture using Orient Anything V2; '
                                             'symmetry and reconstruction errors can cause ambiguity'
                                             if alignment_rotation is not None else
                                             'uncalibrated reconstruction, initially aligned with frozen heading'),
                        'constraints_enforced': False, 'surface_registration_available': False,
                        'collision_checking': False},
        'semantic_context': {'target_object': spec['target_object'], 'action': copy.deepcopy(spec['action']),
                             'constraints': list(spec.get('constraints', [])),
                             'context': spec.get('context', '')}}
    builder = TimelineBuilder(result)
    builder.spawn('ghost_target', position=plan['start'])
    builder.spawn('highlight_target', position=plan['start'])
    if spec['action']['type'] == 'move':
        result['visual_objects'].append({
            'id': 'arrow_motion', 'kind': 'arrow', 'path': [plan['start'], plan['end']],
            'shaft_radius_m': .003, 'head_length_m': .025,
            'appearance': {'color_rgba': [.15, .8, 1., .8], 'opacity': .8}})
        builder.spawn('arrow_motion')
    builder.animate('ghost_target', plan['keyframes'], duration=plan['duration_s'], pivot=plan['pivot'])
    builder.hold('ghost_target', 2., .5)
    for visual in result['visual_objects']:
        builder.visibility(visual['id'], 2.5, False)
    return builder.finish()


def validate_manifest(data):
    """Reject dangling references and incomplete/ambiguous execution data."""
    json.dumps(data, allow_nan=False)
    if data.get('schema_version') != '1.0':
        raise ValueError('unsupported guidance schema_version')

    def ids(items, name):
        values = [item['id'] for item in items]
        if len(values) != len(set(values)):
            raise ValueError(f'duplicate {name} IDs')
        return set(values)

    anchors = ids(data['anchors'], 'anchor')
    assets = ids(data['assets'], 'asset')
    visuals = ids(data['visual_objects'], 'visual')
    for a in data['anchors']:
        uv = finite_vector(a['image_uv'], 2, 'image_uv')
        if np.any(uv < 0) or np.any(uv > 1) or a['point_semantics'] != 'visible_grounded_bbox_center':
            raise ValueError('invalid anchor localization semantics')
    frame = data['coordinate_frame']
    basis = np.column_stack([finite_vector(frame[n+'_world'], 3, n) for n in ('right', 'up', 'forward')])
    if not np.allclose(basis.T@basis, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(basis), 1):
        raise ValueError('invalid frozen frame')
    if not np.allclose(quaternion_matrix(frame['rotation_xyzw']), basis, atol=1e-5):
        raise ValueError('heading quaternion differs from supplied axes')
    if (not np.allclose(basis[:, 1], [0, 1, 0], atol=1e-5)
            or abs(basis[1, 0])+abs(basis[1, 2]) > 1e-5
            or frame['handedness'] != 'Unity_left_handed_y_up_z_forward'):
        raise ValueError('frame must use gravity-aligned Unity axes')
    if not frame['frozen_at_capture'] or frame['units'] != 'meters':
        raise ValueError('coordinate frame must be frozen and metric')

    def pos(value):
        if value['anchor_id'] not in anchors:
            raise ValueError('position references a missing anchor')
        finite_vector(value['offset_m'], 3, 'offset_m')

    for asset in data['assets']:
        if asset['format'] != 'glb' or asset['encoding'] != 'base64':
            raise ValueError('only embedded base64 GLB assets are supported')
        blob = base64.b64decode(asset['data'], validate=True)
        if len(blob) < 20 or blob[:4] != b'glTF' or struct.unpack_from('<II', blob, 4) != (2, len(blob)):
            raise ValueError('invalid embedded GLB')
        if asset['byte_length'] != len(blob) or asset['sha256'] != hashlib.sha256(blob).hexdigest():
            raise ValueError('GLB integrity mismatch')
        size, kind = struct.unpack_from('<II', blob, 12)
        if kind != 0x4E4F534A:
            raise ValueError('GLB JSON chunk missing')
        document = json.loads(blob[20:20+size])
        for entry in document.get('buffers', [])+document.get('images', []):
            if 'uri' in entry and not entry['uri'].startswith('data:'):
                raise ValueError('GLB contains an external dependency')
    kinds = {}
    for visual in data['visual_objects']:
        kinds[visual['id']] = visual['kind']
        color = finite_vector(visual['appearance']['color_rgba'], 4, 'color_rgba')
        if np.any(color < 0) or np.any(color > 1) or visual['appearance']['opacity'] != color[3]:
            raise ValueError('appearance opacity must equal RGBA alpha')
        if visual['kind'] == 'ghost':
            if visual['asset_id'] not in assets:
                raise ValueError('ghost references missing mesh asset')
            alignment = visual['mesh_alignment']
            finite_vector(alignment['position_m'], 3, 'mesh alignment position')
            quaternion_matrix(alignment['rotation_xyzw'])
            if np.any(finite_vector(alignment['scale'], 3, 'mesh scale') <= 0):
                raise ValueError('mesh scale must be positive')
        elif visual['kind'] == 'arrow':
            if len(visual['path']) < 2 or visual['shaft_radius_m'] <= 0 or visual['head_length_m'] <= 0:
                raise ValueError('arrow needs a numerical path and positive dimensions')
            for p in visual['path']:
                pos(p)
        elif visual['kind'] == 'highlight':
            if visual['shape'] != 'wire_sphere' or visual['radius_m'] <= 0:
                raise ValueError('unsupported highlight geometry')
        else:
            raise ValueError('unknown visual primitive')

    spawned = {}
    clips = {}
    holds = {}
    max_end = 0.
    for event in sorted(data['timeline'], key=event_start):
        object_id = event['object_id']
        if object_id not in visuals:
            raise ValueError('timeline references missing visual')
        start = event_start(event)
        if not math.isfinite(start) or start < 0:
            raise ValueError('invalid absolute timeline time')
        max_end = max(max_end, event_end(event))
        if event['type'] == 'spawn':
            if object_id in spawned:
                raise ValueError('visual spawned twice')
            if not isinstance(event['visible'], bool):
                raise ValueError('spawn visible must be boolean')
            spawned[object_id] = start
            quaternion_matrix(event['rotation_xyzw'])
            if kinds[object_id] != 'arrow':
                pos(event['position'])
        elif event['type'] == 'animate_transform':
            if kinds[object_id] == 'arrow':
                raise ValueError('animate_transform applies to ghosts/highlights; arrows have explicit anchored paths')
            duration = event['duration_s']
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('animation duration must be positive')
            if event['position_interpolation'] != 'linear' or event['rotation_interpolation'] != 'slerp_shortest':
                raise ValueError('unsupported interpolation')
            if event['easing'] not in ('linear', 'smoothstep') or event['easing_scope'] != 'each_keyframe_segment':
                raise ValueError('unsupported easing')
            keys = event['keyframes']
            times = [k['time_s'] for k in keys]
            if len(keys) < 2 or times != sorted(set(times)) or times[0] != 0 or times[-1] != duration:
                raise ValueError('keyframes need increasing local times from zero through duration')
            for channel in ('position', 'rotation_xyzw'):
                track = [k for k in keys if channel in k]
                if len(track) < 2 or track[0]['time_s'] != 0 or track[-1]['time_s'] != duration:
                    raise ValueError('each transform channel must span the entire clip')
                for k in track:
                    pos(k[channel]) if channel == 'position' else quaternion_matrix(k[channel])
            pivot = event['pivot']
            if pivot.get('mode') != 'relative_to_animated_position':
                pos(pivot)
            else:
                finite_vector(pivot['offset_m'], 3, 'pivot offset')
            for a, b in clips.get(object_id, []):
                if start < b and start+duration > a:
                    raise ValueError('overlapping transform clips for the same visual are ambiguous; use combined keyframes')
            clips.setdefault(object_id, []).append((start, start+duration))
        elif event['type'] == 'visibility':
            if not isinstance(event['visible'], bool):
                raise ValueError('visibility requires boolean visible')
        elif event['type'] == 'hold':
            holds.setdefault(object_id, []).append((start, start+event['duration_s']))
            if event['duration_s'] < 0 or not math.isfinite(event['duration_s']):
                raise ValueError('invalid hold duration')
        else:
            raise ValueError('unknown timeline operation')
        if event['type'] != 'spawn' and (object_id not in spawned or spawned[object_id] > start):
            raise ValueError('timeline operation occurs before spawn')
    for object_id, intervals in holds.items():
        for a, b in intervals:
            for c, d in clips.get(object_id, []):
                if a < d and b > c:
                    raise ValueError('hold overlaps an active transform for the same visual')
    if visuals != set(spawned):
        raise ValueError('every visual must have an explicit spawn')
    playback = data['playback']
    if not isinstance(playback['loop'], bool):
        raise ValueError('loop must be boolean')
    if playback['duration_s'] < max_end or playback['duration_s'] <= 0:
        raise ValueError('playback duration does not cover timeline')
    if not playback['restart_on_loop'] or not playback['reset_visual_state_on_loop']:
        raise ValueError('loop restart must reset visual state')
    if playback['anchor_resolution'] != 'snapshot_at_playback_start':
        raise ValueError('anchors must be localized to a fixed playback snapshot')
    return data
