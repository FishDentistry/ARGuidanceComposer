# Guidance Composer

Standalone RGB-to-AR guidance composition. Grounding DINO identifies described
objects, rembg prepares the target image, and the existing TripoSR adapter
generates a coarse ghost. The remote composer chooses visuals and animation.
Unity localizes the supplied image anchors and executes numerical instructions.

The primary result is a JSON-compatible Python dictionary with **embedded Base64
GLB geometry**, frozen Unity camera axes, anchors, visuals, and a complete timeline.
It does not depend on local files. An optional FastAPI testing bridge is available under servers/. No Unity client
is implemented here.

## Run

Use the existing environment from this repository:

    .venv310/bin/python -m guidance_composer \
      --image examples/images/apple_test.jpg \
      --spec examples/apple_forward.json examples/apple_right_of.json \
      --output outputs/unity_apple_demo

The optional output directory must be new. It contains guidance.json plus diagnostic
images, GLBs, and preview.png for each specification. Both requests keep the models
loaded. Omit --output to emit a self-contained response on stdout, one JSON object
per line. Image paths belong only to this test harness.

    .venv310/bin/python -m guidance_composer \
      --image TripoSR/examples/chair.png \
      --spec examples/rotate.json examples/move_and_rotate.json \
      --output outputs/unity_chair_demo

Library usage:

    from PIL import Image
    from guidance_composer.grounding import GroundingDino
    from guidance_composer.preprocessing import ObjectPreprocessor
    from guidance_composer.reconstruction import TripoSR
    from guidance_composer.pipeline import GuidanceComposer
    import json

    spec = json.load(open("examples/apple_right_of.json"))
    spec["scene_image"] = Image.open("examples/images/apple_test.jpg").convert("RGB")
    grounder = GroundingDino()
    try:
        composer = GuidanceComposer(grounder, TripoSR(), ObjectPreprocessor())
        response = composer.generate(spec)  # No directory or filesystem output.
        wire_data = json.dumps(response, allow_nan=False)
        # Reuse composer for further requests; calls must be sequential.
    finally:
        grounder.close()

Optional composer.generate(spec, output_directory) writes diagnostics without
adding any paths to the response. Grounding failures raise an explicit error;
with diagnostics enabled they also produce failure.json.

## Input

The minimal input retains camera_pose, target_object, action, constraints, and
context, and now requires camera_axes_world. The API receives scene_image as PIL
image data; example JSONs omit it.

    {
      "camera_pose": {
        "position": [0.0, 1.6, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0]
      },
      "camera_axes_world": {
        "right": [1.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0]
      },
      "target_object": "yellow apple",
      "action": {
        "type": "move",
        "qualifiers": {
          "placement": "right_of",
          "reference_object": "Closed silver laptop with stickers on its lid",
          "distance": 0.10,
          "unit": "meters"
        }
      },
      "constraints": ["Keep the apple on the tabletop"],
      "context": "Move the apple beside the laptop"
    }

Object descriptions are natural-language grounding prompts. No preassigned object
IDs, 3D object positions, semantic object frames, or visualization choices are
required. Generated anchor/asset/visual IDs exist only in the response.

The supplied axes use Unity world coordinates: +X camera-right, +Y world-up, +Z
horizontal camera-forward. They must be finite, unit length, mutually orthogonal,
gravity aligned, and satisfy right = cross(up, forward). The parser checks the
heading against the captured quaternion [x,y,z,w]. Pitch/roll do not tilt horizontal
movement. Nearly vertical poses can use independently reliable supplied axes;
consistency is then explicitly marked uncheckable. Missing/bad axes are rejected.

axes_from_pose(camera_pose) is available to capture producers and example authors.
It projects Unity local +Z onto the horizontal plane and fails explicitly near
vertical. It is not a substitute for supplied axes at generation time.
All relevant examples contain axes; translate_pitched.json demonstrates nonidentity
yaw, pitch, and roll with correctly derived axes. The frame is copied at capture
and never depends on later headset movement.

This replaces the previous right-handed/−Z-forward convention. See
[Unity's coordinate and quaternion conventions](https://docs.unity.com/en-us/engine/6000.0/manual/working-with-gameobjects/gameobject-fundamentals/rotation-orientation/quaternion-and-euler-rotations-in-unity).
Old output directories retain their historical conventions and should be regenerated.

## Spatial semantics

- Translation uses direction and an offset from the target's original anchor:
  25 cm right produces endpoint anchor_target + [0.25,0,0].
- Placement uses placement and reference_object: 10 cm right_of the laptop
  produces endpoint anchor_reference + [0.10,0,0].
- right/left, up/down, forward/backward map to ±X, ±Y, ±Z of the frozen frame.
  behind is +Z (away); in_front_of is −Z (toward).
- Placement distance is **anchor-to-anchor**, never boundary clearance.
  Laptop dimensions or orientation are not needed. distance_mode from the earlier
  boundary-based prototype is rejected rather than silently reinterpreted.
- References are grounded only when a placement needs one. Surface/constraint
  text is retained but does not trigger additional grounding or orientation models.
- Both move and rotate are supported. A move can optionally include a semantic
  qualifiers.rotation object for simultaneous motion. Rotate qualifiers require
  angle, unit degrees, and explicit axis x/y/z. Y rotations use clockwise or
  counterclockwise viewed from above. Other axes require viewing_convention
  from_positive_axis or from_negative_axis.
- Optional pivot_offset_m specifies a pivot relative to the animated target
  position, in frozen camera meters. Lower-level timeline composition can also
  anchor a pivot to another localized object. These are physical-action qualifiers,
  not visualization instructions.

Unity Quaternion.AngleAxis numerical conventions are used: +90 degrees about +Y
takes +Z toward +X (clockwise from above). With no direction, signed angles use
that numerical convention. Quaternions are [x,y,z,w] orientations in the frozen
heading frame. The composer supplies actual keyframes; Unity does not derive
angles, choose pivots, or interpret directional words.

## Output and timeline

The complete execution contract is documented in
[docs/unity_guidance_contract.md](docs/unity_guidance_contract.md).
manifest.py builds and validates responses; playback.py is a Python reference
executor for deterministic tests and previews.

Responses contain:

- coordinate_frame: frozen world axes, heading quaternion, units, validation status.
- capture: captured pose and source-image dimensions/UV convention.
- anchors: normalized image points with visible_grounded_bbox_center semantics.
- assets: self-contained Base64 GLBs, integrity metadata, explicit Unity import policy.
- visual_objects: ghost alignment/appearance, procedural arrow paths and highlights.
- timeline: spawn, transform keyframes, hold, and visibility operations.
- playback: duration, loop/reset behavior, anchor-localization snapshot policy.
- limitations and semantic_context: diagnostics, not instructions Unity must parse.
- timings_ms: grounding, preprocessing, reconstruction, extraction, composition,
  and optional diagnostic export measurements.

The default composition shows a translucent ghost and target wire-sphere highlight.
Moves also receive an anchored arrow. The ghost holds initially for 0.5 s, animates
for 1.5 s, holds its endpoint for 0.5 s, hides for 0.25 s, then restarts.
Appearance explicitly overrides material color/alpha rather than multiplying alpha.

TimelineBuilder supports remote-side sequential or overlapping operations on
multiple visuals, shared mesh assets, fixed or moving pivots, linear/smoothstep
interpolation, quaternion slerp, visibility, holds, and looping. Large rotations
are subdivided into numerical quaternion steps of at most 90 degrees, preserving
the intended turn despite shortest-path interpolation. Conflicting simultaneous
transform clips for the same visual are rejected; use combined tracks instead.
The semantic input still describes one physical action; more elaborate sequences
can be composed through this lower-level API without adding primitive fields
to the guidance specification.

## Localization versus mesh accuracy

Assume Unity accurately maps every supplied image anchor into its physical world.
Position offsets remain exact meters even when ghost shape/scale are approximate.
When keyframes reference different anchors, resolve their world positions first
and interpolate those resolved positions. Reference intrinsic orientation never
enters that calculation.

TripoSR mesh-local units are not meters. Assets retain reconstructed axes and are
centered before packaging. mesh_alignment supplies an explicit display scale
(maximum ghost extent 0.12 m), rotation, and position. This is a presentation policy,
not an estimate of the physical object's dimensions or orientation. A future
alignment estimator can replace it; Unity has no alignment choices to make.

GLB is right-handed. The contract explicitly requires a once-only Z reflection
for vertices/normals and reversed triangle winding at import into Unity. An
importer that already performs conversion must be configured/adapted to match
this policy; do not apply it twice. Instruction positions/quaternions are already
Unity coordinates and must not receive this asset conversion.

No collision checking, tabletop-plane recovery, constraint solver, articulated
reconstruction, or physical orientation estimation is implemented. Thus an exact
anchor-relative destination can overlap a laptop or fail to rest on a table.
Constraints are preserved with an explicit not-enforced status.

## Previews and diagnostics

preview.png evaluates the same manifest and embedded GLB used by Unity. It shows
initial/final ghosts, the supplied arrow/highlight, captured directions, image
anchors, and a separate camera context panel. The target is at origin in a
preview-only layout; reference anchors are placed from normalized image offsets
using an illustrative 0.6 m image span. This is not measured localization.

The manifest remains unchanged by those preview assumptions.
preview_anchors.json records the illustrative positions. initial_ghost.glb,
transformed_ghost.glb, and demonstration.glb use that same illustrative execution;
ghost.glb is the embedded asset. reconstructed.glb preserves the centered original
mesh. All diagnostic GLBs are converted back to glTF coordinates for standard
viewers. No preview file is needed by the remote response.

## Dependencies and verification

The existing .venv310, TripoSR source, CPU marching cubes, PyTorch/CUDA, and cached
weights are unchanged. Resolution remains 64 and chunk size 4096. Grounding DINO's
newer Transformers dependencies remain isolated in .guidance-deps and its persistent
subprocess; no new packages were added for this update.

If recreating that project-local grounding dependency directory:

    .venv310/bin/python -m pip install --target .guidance-deps --no-deps \
      transformers==4.40.2 tokenizers==0.19.1 huggingface-hub==0.23.5

Weights for IDEA-Research/grounding-dino-tiny, stabilityai/TripoSR, its DINO encoder,
and rembg's u2net.onnx must already be cached. The CLI uses offline Hugging Face
mode. Grounding failures and competing detections remain explicit.

    .venv310/bin/python -m unittest discover -s tests -v

Tests cover supplied axes/pose consistency, pitch/roll, vertical captures,
frozen frames, exact anchor offsets, mesh-independent metric movement, rotations,
pivots, combined motion, long-turn interpolation, different-anchor keyframes,
sequences/overlaps, visibility/loops, embedded GLB integrity, serialization,
reference validation, and generation without filesystem output.

Real Grounding DINO + TripoSR translation/placement manifests and previews were
generated under outputs/unity_guidance_2026_10_21/. These are diagnostic locations,
not dependencies of the response. Refer to each guidance.json for the complete
serializable result.

Validated on October 21, 2026: all 28 tests pass. Real model runs succeeded
for apple translation, apple-to-laptop anchor placement, chair rotation, and
simultaneous chair translation/rotation. Every saved guidance.json passed
manifest validation, embedded-asset decoding, and numerical endpoint evaluation.
Chair outputs are under outputs/unity_rotation_2026_10_21/.

## FastAPI testing server

Unity can POST a Base64 scene image and captured camera pose to /guidance.
The server fills the semantic fields from a selected examples/ JSON, keeps models
loaded, and returns the complete embedded-asset manifest. See
[servers/README.md](servers/README.md) for the payload and test sender.

    .venv310/bin/python -m uvicorn servers.app:app --host 0.0.0.0 --port 8000 --workers 1

Set GUIDANCE_EXAMPLE=apple_right_of.json before starting to select that scenario.
