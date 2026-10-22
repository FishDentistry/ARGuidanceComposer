"""Optional diagnostics. All geometry and execution data come from the manifest."""
import base64
import io
import json
from pathlib import Path
import numpy as np
import trimesh
from .frames import quaternion_matrix
from .playback import evaluate, heading_basis

REFLECT_Z = np.diag([1., 1., -1., 1.])


def annotation_font(width):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('DejaVuSans.ttf', max(12, width//110))
    except OSError:
        return ImageFont.load_default()


def ghost_mesh(mesh):
    ghost = mesh.copy()
    ghost.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name='Guidance ghost', baseColorFactor=[51, 179, 255, 102],
        metallicFactor=0., roughnessFactor=.8, alphaMode='BLEND', doubleSided=True))
    return ghost


def illustrative_anchors(manifest):
    """Preview-only layout: target at origin, image offsets mapped onto a flat plane.

    This is not localization; Unity supplies the accurate physical anchor positions.
    """
    basis = heading_basis(manifest)
    target_uv = np.array(manifest['anchors'][0]['image_uv'])
    anchors = {}
    for anchor in manifest['anchors']:
        du, dv = np.array(anchor['image_uv'])-target_uv
        anchors[anchor['id']] = basis @ np.array([du*.6, 0., -dv*.6])
    return anchors


def ghost_at(manifest, anchors, time):
    visual = next(v for v in manifest['visual_objects'] if v['kind'] == 'ghost')
    asset = next(a for a in manifest['assets'] if a['id'] == visual['asset_id'])
    mesh = trimesh.load(io.BytesIO(base64.b64decode(asset['data'])), file_type='glb', force='mesh')
    # Match the contract's once-only glTF-to-Unity import conversion.
    mesh.apply_transform(REFLECT_Z)
    alignment = visual['mesh_alignment']
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(alignment['rotation_xyzw']) @ np.diag(alignment['scale'])
    transform[:3, 3] = alignment['position_m']
    mesh.apply_transform(transform)
    state = evaluate(manifest, anchors, time)[visual['id']]
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(state['rotation_xyzw_world'])
    transform[:3, 3] = state['position_world_m']
    mesh.apply_transform(transform)
    return mesh


def export_diagnostics(manifest, output):
    output = Path(output)
    anchors = illustrative_anchors(manifest)
    clip = next(e for e in manifest['timeline'] if e['type'] == 'animate_transform')
    end = clip['start_time_s']+clip['duration_s']
    initial, final = ghost_at(manifest, anchors, 0), ghost_at(manifest, anchors, end)
    # Files are glTF RH for standard viewers; positions convert back from Unity.
    start_glb, end_glb = initial.copy(), final.copy()
    start_glb.apply_transform(REFLECT_Z)
    end_glb.apply_transform(REFLECT_Z)
    start_glb.export(output/'initial_ghost.glb')
    end_glb.export(output/'transformed_ghost.glb')
    scene = trimesh.Scene()
    scene.add_geometry(start_glb, node_name='initial')
    scene.add_geometry(end_glb, node_name='transformed')
    scene.export(output/'demonstration.glb')
    (output/'ghost.glb').write_bytes(base64.b64decode(manifest['assets'][0]['data']))
    (output/'preview_anchors.json').write_text(json.dumps({
        'localization': 'illustrative_only_not_measured',
        'anchors_world_m': {k: v.tolist() for k, v in anchors.items()}}, indent=2))
    preview(manifest, anchors, output/'preview.png', initial, final)


def preview(manifest, anchors, path, initial=None, final=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    clip = next(e for e in manifest['timeline'] if e['type'] == 'animate_transform')
    end = clip['start_time_s']+clip['duration_s']
    initial = initial if initial is not None else ghost_at(manifest, anchors, 0)
    final = final if final is not None else ghost_at(manifest, anchors, end)
    fig = plt.figure(figsize=(14, 7))
    camera = np.array(manifest['capture']['camera_pose']['position'])
    frame = heading_basis(manifest)
    states = evaluate(manifest, anchors, 0)
    detail_points = [initial.vertices, final.vertices, *[v[None, :] for v in anchors.values()]]
    ghost_id = next(v['id'] for v in manifest['visual_objects'] if v['kind'] == 'ghost')
    path_points = np.array([evaluate(manifest, anchors, t)[ghost_id]['position_world_m']
                           for t in np.linspace(clip['start_time_s'], end, 31)])
    detail_points.append(path_points)
    for panel in (1, 2):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        for mesh, color, alpha in [(initial, '#909090', .65), (final, '#16cbe6', .4)]:
            ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces][:, :, [0, 2, 1]],
                                                facecolor=color, alpha=alpha, linewidth=0))
        ax.plot(path_points[:, 0], path_points[:, 2], path_points[:, 1], '--', color='#16a6bb')
        for anchor in manifest['anchors']:
            p = anchors[anchor['id']]
            ax.scatter(*p[[0, 2, 1]], color='#7660aa', s=20)
            ax.text(*p[[0, 2, 1]], ' '+anchor['id'], fontsize=7)
        for visual in manifest['visual_objects']:
            if visual['kind'] == 'arrow':
                points = states[visual['id']]['path_world_m']
                for a, b in zip(points, points[1:]):
                    vector = b-a
                    ax.quiver(*a[[0, 2, 1]], *vector[[0, 2, 1]], color='#db8500', arrow_length_ratio=.12)
            elif visual['kind'] == 'highlight':
                p = states[visual['id']]['position_world_m']
                radius = visual['radius_m']
                theta = np.linspace(0, 2*np.pi, 65)
                for plane in ((0, 1), (0, 2), (1, 2)):
                    ring = np.tile(p, (len(theta), 1))
                    ring[:, plane[0]] += np.cos(theta)*radius
                    ring[:, plane[1]] += np.sin(theta)*radius
                    ax.plot(ring[:, 0], ring[:, 2], ring[:, 1], color='#d7b530', alpha=.45, linewidth=.6)
        for i, (name, color) in enumerate([('right', '#d43c3c'), ('up', '#32a852'), ('forward', '#db8500')]):
            axis = frame[:, i]*.10
            ax.quiver(0, 0, 0, *axis[[0, 2, 1]], color=color, arrow_length_ratio=.15)
            ax.text(*axis[[0, 2, 1]], name, fontsize=7, color=color)
        points = np.vstack(detail_points+([camera[None, :]] if panel == 2 else []))[:, [0, 2, 1]]
        if panel == 2:
            ax.scatter(*camera[[0, 2, 1]], color='#b22a63', s=40)
            ax.text(*camera[[0, 2, 1]], ' captured camera', color='#b22a63', fontsize=8)
        center = (points.min(0)+points.max(0))/2
        radius = max(float(np.ptp(points, axis=0).max())*.65, .16)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set(xlabel='Unity X (m)', ylabel='Unity Z (m)', zlabel='Unity Y (m)',
               title='Guidance detail' if panel == 1 else 'Capture context')
    fig.suptitle('Manifest playback — gray: initial; cyan: final; orange: supplied arrow\n'
                 'Illustrative anchor localization and mesh scale; numerical offsets are exact meters', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
