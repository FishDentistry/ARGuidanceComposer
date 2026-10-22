# Guidance Composer

A standalone prototype for composing augmented-reality guidance from a scene image,
captured camera pose, and structured task specification. The output is a
self-contained JSON manifest that Unity can localize and play back.

The pipeline uses **Grounding DINO → SAM2.1 → TripoSR → Orient Anything V2** to locate
the target, segment it, reconstruct a ghost mesh, and estimate its initial
orientation. The composer supplies the movement path, arrow, highlight, animation,
and playback instructions. Unity supplies physical anchor localization and rendering.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Command-line usage](#command-line-usage)
- [Testing server](#testing-server)
- [Guidance specifications](#guidance-specifications)
- [Coordinate conventions](#coordinate-conventions)
- [Python API](#python-api)
- [Output and Unity integration](#output-and-unity-integration)
- [Diagnostics](#diagnostics)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)

## Requirements

The supported development setup is Linux with Python 3.10, Git, and an NVIDIA GPU
with a compatible driver and CUDA-enabled PyTorch. Headless EGL/OpenGL support is
required for Orient Anything's mesh-reference rendering; ModernGL is the Python
binding. A working CUDA installation alone does not establish EGL support.


Network access is required during installation and model download. Runtime uses
local checkpoints and cached Hugging Face files. The orientation checkpoint alone
is approximately 5.05 GB; additional space is required for the other models,
PyTorch, and generated artifacts.

## Installation

Clone the repository and enter its root directory before running these commands.
The steps below describe a fresh installation.

### 1. Create the Python environment

```sh
python3.10 -m venv .venv310
.venv310/bin/python -m pip install --upgrade pip setuptools wheel
```

Commands use the environment's Python explicitly; shell activation is optional.

### 2. Install PyTorch and the main dependencies

Install a matching PyTorch/TorchVision pair for the host GPU and driver. SAM2
requires PyTorch 2.5.1 or later and TorchVision 0.20.1 or later. One compatible
CUDA 12.1 pair is:

```sh
.venv310/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Use the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/)
when another CUDA build is needed. Retain an existing compatible pair rather than
replacing it solely to match the example above.

Install the main environment:

```sh
SAM2_BUILD_CUDA=0 .venv310/bin/python -m pip install --no-build-isolation -r requirements.txt
```

`requirements.txt` includes reconstruction, segmentation, orientation rendering,
server, preview, and test dependencies. `SAM2_BUILD_CUDA=0` skips SAM2's optional
custom CUDA extension; model inference still uses PyTorch CUDA. 


### 3. Install the isolated Grounding DINO dependencies

```sh
.venv310/bin/python -m pip install --target .guidance-deps --no-deps -r requirements-grounding.txt
```

This is the only separate dependency list. TripoSR uses Transformers 4.35.0 and
Tokenizers 0.14.1, while the Grounding DINO API requires a newer stack.
`requirements-grounding.txt` supplies Transformers 4.40.2, Tokenizers 0.19.1, and
Hugging Face Hub 0.23.5 to the grounding subprocess. The remaining packages,
including PyTorch, are shared from `.venv310`.

**Do not install `requirements-grounding.txt` directly into `.venv310` or include it
in the main requirements file.** The worker prepends `.guidance-deps` to its import
path; the main process continues using its compatible reconstruction dependencies.

### 4. Install the TripoSR source and marching-cubes patch

```sh
git clone https://github.com/VAST-AI-Research/TripoSR.git TripoSR
git -C TripoSR checkout 107cefdc244c39106fa830359024f6a2f1c78871
git -C TripoSR apply --check ../patches/triposr-cpu-marching-cubes.patch
git -C TripoSR apply ../patches/triposr-cpu-marching-cubes.patch
```

The patch reproduces this project's scikit-image marching-cubes implementation.
Mesh extraction runs on CPU and returns tensors to the reconstruction device;
TripoSR inference remains on the selected device. This avoids building
`torchmcubes`.

Do not install `TripoSR/requirements.txt` after the consolidated requirements:
that upstream list includes dependencies and version choices for other TripoSR
applications. If the patch is already applied, the forward apply check fails;
`git -C TripoSR apply --reverse --check ../patches/triposr-cpu-marching-cubes.patch`
can verify that state without changing files.

### 5. Download model files

The composer does not download models on its first request. Populate the Hugging
Face cache and download SAM2 before starting the CLI or server:

```sh
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 .venv310/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download("IDEA-Research/grounding-dino-tiny")
for filename in ("config.yaml", "model.ckpt"):
    hf_hub_download("stabilityai/TripoSR", filename)
hf_hub_download("facebook/dino-vitb16", "config.json")

checkpoint = Path("checkpoints/sam2.1_hiera_small.pt")
checkpoint.parent.mkdir(parents=True, exist_ok=True)
if not checkpoint.exists():
    temporary = checkpoint.with_suffix(".pt.part")
    urlretrieve(
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        temporary,
    )
    temporary.replace(checkpoint)
PY

.venv310/bin/python -m servers.setup_orientation
```

The orientation setup command retrieves the pinned official source into
`third_party/orient_anything_v2/` and the checkpoint into
`checkpoints/orient_anything_v2/`. It verifies the source revision and checkpoint
SHA-256. These are source/model assets, not another Python environment. Both
locations are ignored by Git.

| Component | Runtime model/source |
|---|---|
| Grounding | `IDEA-Research/grounding-dino-tiny`, in the Hugging Face cache |
| Reconstruction | `stabilityai/TripoSR` configuration and checkpoint, in the cache |
| Reconstruction image encoder | `facebook/dino-vitb16/config.json`, in the cache |
| Segmentation | `checkpoints/sam2.1_hiera_small.pt` |
| Orientation | `checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt` and the pinned source checkout |

Keep the same Hugging Face cache available to the account running the server.
See [segmentation setup](docs/segmentation.md) and
[orientation setup](docs/orientation.md) for checkpoint details and overrides.

### 6. Verify CUDA, EGL, and imports

```sh
.venv310/bin/python - <<'PY'
import torch
import moderngl
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
context = moderngl.create_standalone_context(backend="egl")
print("EGL renderer:", context.info["GL_RENDERER"])
context.release()
PY
```

A complete CLI or HTTP request is the final model-loading check. The unit tests
use lightweight model substitutes and do not establish that all checkpoints are
installed.

## Command-line usage

Supply an image and one or more specification files:

```sh
.venv310/bin/python -m guidance_composer \
  --image examples/images/apple_test.jpg \
  --spec examples/apple_forward.json examples/apple_right_of.json \
  --output outputs/apple_demo
```

The output directory must not already exist. Each specification receives a
numbered subdirectory, such as `outputs/apple_demo/00_apple_forward/`, containing
`guidance.json`, `preview.png`, and the diagnostic artifacts described below.
Models load once and are reused across specifications.

Omit `--output` to emit the complete response as one JSON object per line (NDJSON
for multiple specifications). In that mode no diagnostic output directory is
created. Errors go to stderr and any failed specification produces a nonzero exit
status.

Common options:

| Option | Purpose |
|---|---|
| `--image` | Required scene-image path; the CLI loads it into memory |
| `--spec` | Required specification path or paths |
| `--output` | New parent directory for diagnostic runs |
| `--device` | Model device; default `cuda:0` |
| `--threshold` | Grounding box threshold; default `0.3` |
| `--grounding-model`, `--reconstruction-model` | Override Hugging Face model identifiers |
| `--sam2-checkpoint`, `--sam2-config` | Matching SAM2 checkpoint/configuration pair |
| `--orientation-checkpoint`, `--orientation-source` | Override Orient Anything V2 asset paths |

Run `python -m guidance_composer --help` using the environment's Python for the
full option list. Alternative Hugging Face models must also be cached in advance.

## Testing server

Start the FastAPI bridge with an explicit semantic specification:

```sh
GUIDANCE_EXAMPLE=pop_can_laptop_test.json \
.venv310/bin/python -m uvicorn servers.app:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

Wait for `Application startup complete`. Use one worker so the process holds one
copy of each model. Inference is serialized on a dedicated model thread; health
requests remain responsive during processing. The server is an unauthenticated
testing endpoint.

| Endpoint | Purpose |
|---|---|
| `POST /guidance` | Compose guidance and return the complete JSON manifest |
| `GET /health` | Report readiness, selected example, device, and serialization behavior |
| `GET /docs` | Interactive API documentation |
| `GET /openapi.json` | Request schema |

Unity sends only the captured image and camera metadata:

```json
{
  "scene_image_base64": "<raw Base64-encoded PNG or JPEG file bytes>",
  "camera_pose": {
    "position": [0.0, 1.6, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0]
  }
}
```

Vector/quaternion values are arrays, not `{x, y, z}` objects. The image must contain
encoded image-file bytes, not a raw pixel buffer or a `data:image/...` URL. Capture
the image and pose at the same instant. Optional `camera_axes_world` has the same
structure as the specification field below; it is required for a near-vertical
camera whose horizontal heading cannot be derived reliably from its pose.

The server loads `target_object`, `action`, `constraints`, and `context` from the
selected file in `examples/`. It replaces that file's image and **all camera
metadata** with the request capture. Supplied capture axes are validated; otherwise
the server derives them from the incoming pose. It never borrows the example's
heading for a new capture.

Edits to the selected example take effect on the next request. Changing the
selected filename, model configuration, or Python code requires a server restart.
`GET /health` identifies the active example. The environment default is currently
`pop_can_test.json`; an explicit `GUIDANCE_EXAMPLE` avoids relying on that default.

Each processed capture creates a unique directory under the repository's
`outputs/`, named `server_<example>_<UTC timestamp>_<id>/`. The saved `guidance.json`
matches the HTTP response. Unity receives embedded assets directly and does not
fetch files from that directory. Requests rejected before composition do not
create artifacts; failures during generation retain available intermediates and
`failure.json`.

Test the same protocol without Unity:

```sh
.venv310/bin/python -m servers.send_test \
  --image examples/images/apple_test.jpg \
  --camera-spec examples/apple_forward.json \
  --url http://127.0.0.1:8000/guidance \
  --output /tmp/guidance_response.json
```

Use a capture containing the objects named by the server's selected example.
`--camera-spec` supplies **only camera metadata**, not the semantic task; it does
not change `GUIDANCE_EXAMPLE`. `--include-axes` also sends that file's axes. A Unity
device on another machine must use the server's reachable address, not `0.0.0.0`
or the device's own `localhost`.

The image limit is 24 MiB of encoded-file data and 40 megapixels after decoding.
PNG, JPEG, and the first frame of JPEG-compatible MPO files are accepted. JPEG EXIF
orientation is applied before grounding; returned UVs refer to that oriented image.
The test sender's default timeout is 180 seconds; Unity must allow time for model
inference, diagnostic export, and any queued requests.

See [server configuration and errors](servers/README.md) for environment variables,
checkpoint overrides, and HTTP error details.

## Guidance specifications

A specification describes the target and requested transformation. It does not
contain a mesh, object IDs, or Unity animation commands. JSON examples omit the
image because the CLI or HTTP capture supplies it separately.

A complete translation specification:

```json
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
  "target_object": "gray soda can",
  "action": {
    "type": "move",
    "qualifiers": {
      "direction": "right",
      "distance": 0.2,
      "unit": "meters"
    }
  },
  "constraints": ["Keep the can supported on the tabletop"],
  "context": "Move the can 20 cm to the captured camera's right."
}
```

The camera values above illustrate an identity camera orientation. Real captures
must provide their actual pose and matching axes.

### Fields and execution authority

| Field | Meaning |
|---|---|
| `target_object` | Required, nonempty natural-language description used to ground the target |
| `camera_pose.position` | Required three-element Unity-world position in meters |
| `camera_pose.rotation` | Required unit quaternion in `[x, y, z, w]` order |
| `camera_axes_world` | Required by the direct Python/CLI specification; frozen, gravity-aligned world-space right/up/forward vectors |
| `action.type` | `move` or `rotate` |
| `action.qualifiers` | Numerical and enumerated fields that determine execution |
| `constraints` | Optional list of strings; retained as metadata and **not enforced** |
| `context` | Optional descriptive string; retained as metadata and **not interpreted as an instruction** |

**Execution follows `action.qualifiers`.** Neither `context` nor `constraints`
overrides its direction, distance, reference, rotation, or pivot. If qualifiers say
`distance: 0.1` while context says “20 cm,” the operation uses **10 cm**. There is
no natural-language consistency checker or constraint solver.

Descriptions should identify a visible, distinguishable object. Grounding rejects
no-match cases and similarly scored competing detections instead of selecting one
arbitrarily. A reference object is grounded only when a placement requires it;
mentioning an object in context, constraints, or a `surface` string does not add a
reference anchor.

### Translation: move from the target's starting position

Use `direction` for a displacement from the target's own initial anchor:

```json
{
  "type": "move",
  "qualifiers": {
    "direction": "right",
    "distance": 0.2,
    "unit": "meters"
  }
}
```

This moves the can **20 cm right from wherever it starts**. It does not position
the can relative to a laptop or another object.

| `direction` | Offset in frozen captured-heading coordinates |
|---|---|
| `right` | `+X` |
| `left` | `−X` |
| `up` | `+Y` |
| `down` | `−Y` |
| `forward`, `behind` | `+Z`, away along the captured horizontal heading |
| `backward`, `in_front_of` | `−Z`, toward the captured camera along that heading |

Distance must be finite and nonnegative. The only supported length unit is
`meters`: `0.1` is 10 cm and `0.2` is 20 cm. Use the opposite direction instead
of a negative distance. Zero distance is allowed.

For target anchor position `T`, frozen basis `F`, unit direction `d`, and distance
`s`, the destination is `T + F × (s d)`.

### Placement: end at an offset from a reference anchor

Use `placement` and `reference_object` for a destination defined relative to
another object:

```json
{
  "type": "move",
  "qualifiers": {
    "placement": "right_of",
    "reference_object": "dark gray open laptop",
    "distance": 0.1,
    "unit": "meters"
  }
}
```

This means **end 10 cm to the captured camera's right of the laptop's anchor**.
It does not mean “move the can right by 10 cm.”

| `placement` | Destination offset from the reference anchor |
|---|---|
| `right_of` | `+X × distance` |
| `left_of` | `−X × distance` |
| `behind` | `+Z × distance` |
| `in_front_of` | `−Z × distance` |

For reference anchor position `R`, the destination is `R + F × (s d)`. The
animation and arrow run from the target's starting anchor to that destination.
The reference's full world position is used, so the path can also change depth
or height.

For example, along the captured right axis, let the laptop anchor be at 0 cm and
the placement destination at +10 cm:

| Can's starting position | Movement needed to end at +10 cm |
|---|---|
| −20 cm | 30 cm right |
| +5 cm | 5 cm right |
| +30 cm | 20 cm left |
| +10 cm | No displacement along that axis |

A leftward arrow is therefore valid for a `right_of` placement when the target
starts farther right than its destination.

**Offsets are measured between anchors, not from object edges.** An anchor is the
normalized image-bounding-box center, localized into the physical world by Unity.
It is not a known mesh center, center of mass, support point, or rightmost edge.
Ten centimeters right of a wide laptop's anchor can still lie within the laptop's
physical footprint. No boundary clearance or collision avoidance is calculated.

### Translation and placement are mutually exclusive

A `move` action requires exactly one of `direction` and `placement`.
`reference_object` is required for placement. Adding it to a directional move
does not change the origin of that move or cause the object to be grounded.

| Intended task | Specification |
|---|---|
| Move the can right by 20 cm | `direction: "right"`, `distance: 0.2` |
| Put the can 10 cm right of the laptop anchor | `placement: "right_of"`, laptop reference, `distance: 0.1` |
| Leave 10 cm of empty space beyond the laptop's right edge | Not supported; physical boundaries are unavailable |
| Move radially away from the laptop's current position | Not supported as a reference-relative direction |
| Use the laptop's own front/right axes | Not supported; spatial terms use the captured camera heading |

### Rotation

A pure rotation uses `type: "rotate"`:

```json
{
  "type": "rotate",
  "qualifiers": {
    "axis": "y",
    "angle": 90,
    "unit": "degrees",
    "direction": "clockwise"
  }
}
```

Angles must be finite, and the only angular unit is `degrees`. Axes are the frozen
captured-heading `x`, `y`, and `z` axes, not the object's estimated intrinsic axes.
A Y-axis clockwise rotation is viewed from above and turns heading `+Z` toward
heading `+X`. The default pivot is the target's starting anchor.

When `direction` is present, use a nonnegative angle and either `clockwise` or
`counterclockwise`. Without `direction`, the angle is signed and follows Unity
AngleAxis. The numerical positive-angle mappings are:

| Axis | Effect of +90° |
|---|---|
| `x` | `+Y` turns toward `+Z` |
| `y` | `+Z` turns toward `+X` |
| `z` | `+X` turns toward `+Y` |

Nonvertical rotations require `viewing_convention: "from_positive_axis"` or
`"from_negative_axis"`. With the latter, the compiled angle sign is reversed.
For Y, omit the convention or use `from_above` / `from_positive_axis`.
Rotations larger than 180° are split into intermediate keyframes so interpolation
preserves the requested turn rather than taking a shorter reverse rotation.

`pivot_offset_m` optionally specifies `[x, y, z]` in meters in the frozen heading
frame, relative to the animated target-anchor position. It is not an offset in
mesh-local axes. An off-center pivot can make the mesh root travel along an arc,
even for a pure rotation.

### Combined movement and rotation

Nest a rotation inside a move action's qualifiers:

```json
{
  "type": "move",
  "qualifiers": {
    "direction": "right",
    "distance": 0.25,
    "unit": "meters",
    "rotation": {
      "axis": "y",
      "angle": 90,
      "unit": "degrees",
      "direction": "clockwise",
      "pivot_offset_m": [0.04, 0.0, 0.0]
    }
  }
}
```

Movement and rotation occur simultaneously. The example uses a pivot 4 cm right
of the moving base position. With a zero pivot offset, the root follows the
specified translation while the mesh rotates about that root. With an off-center
pivot, the root receives the additional rotational displacement; the arrow still
represents the base translation. Placement can also contain a nested rotation.

### Unsupported and descriptive qualifiers

`relative_to` may be omitted or set to `"camera"`; world-relative and object-relative
values are rejected. `distance_mode` is obsolete and rejected. A `surface` string,
if present, must be a nonempty description, but does not establish a plane,
reference anchor, or support constraint. Unknown fields are not an extension API;
use only the fields documented here.

## Coordinate conventions

All physical poses and instruction transforms use Unity's **Y-up, +Z-forward**
world convention. Quaternions use `[x, y, z, w]`, and lengths use meters. World
origin is arbitrary; the captured camera and Unity-localized anchors must share
that world frame.

The instruction basis is frozen at capture:

- `up` is world `[0, 1, 0]`.
- `forward` is the camera's forward direction projected onto the horizontal plane.
- `right` is `cross(up, forward)`.

The supplied vectors must be unit length, mutually orthogonal, gravity aligned,
and consistent with the captured camera quaternion. A near-vertical camera needs
an explicitly supplied reliable horizontal heading. The instruction frame does
not follow subsequent headset motion, camera roll, or an object's inferred facing.
“Forward” therefore means horizontal captured-camera forward, not the laptop's
front, the image's vertical direction, or the camera's pitched optical ray.

Orient Anything uses the full captured camera quaternion, including pitch and
roll, to estimate the mesh's initial appearance in the world. This changes
`mesh_alignment`, not the meaning of move directions or rotation axes.

## Python API

```python
import json
from PIL import Image
from guidance_composer.grounding import GroundingDino
from guidance_composer.orientation import OrientAnythingV2
from guidance_composer.pipeline import GuidanceComposer
from guidance_composer.preprocessing import ObjectPreprocessor
from guidance_composer.reconstruction import TripoSR

spec = json.load(open("examples/apple_forward.json"))
spec["scene_image"] = Image.open("examples/images/apple_test.jpg").convert("RGB")

grounder = GroundingDino("cuda:0")
try:
    composer = GuidanceComposer(
        grounder,
        TripoSR("cuda:0"),
        ObjectPreprocessor("cuda:0"),
        OrientAnythingV2("cuda:0"),
    )
    response = composer.generate(spec)
    # Optional diagnostic export: composer.generate(spec, "outputs/new_run")
finally:
    grounder.close()
```

The direct API accepts `scene_image` as an in-memory PIL image and requires both
camera pose and axes. Reuse the composer sequentially to retain loaded models;
its predictors are stateful and not designed for concurrent calls. Omitting the
fourth constructor argument disables orientation estimation explicitly; the
manifest then marks mesh orientation as uncalibrated.

## Output and Unity integration

The primary result is a JSON-compatible dictionary, returned directly by the API
and `POST /guidance`, or saved as `guidance.json`. It contains:

| Field | Content |
|---|---|
| `coordinate_frame`, `capture` | Frozen heading, captured camera pose, source-image dimensions and UV convention |
| `anchors` | Response-local IDs, descriptions, normalized UV centers, and diagnostic bounding boxes |
| `assets` | Base64-encoded GLB geometry, byte lengths, hashes, and import instructions |
| `visual_objects` | Ghost mesh alignment, appearance, highlight, and movement arrow |
| `timeline`, `playback` | Numerical keyframes, pivots, visibility events, timing, and loop rules |
| `limitations`, `semantic_context`, `timings_ms` | Accuracy flags, retained task metadata, and timing diagnostics |

Unity resolves each image anchor into a world position. Image UVs have a top-left
origin, with U increasing rightward and V downward. Instruction distances become
exact numerical meter offsets from those localized anchors; the composer does not
recover metric scene geometry from the RGB image.

GLB assets use right-handed geometry. The contract specifies one Z reflection for
vertices/normals and a triangle-winding reversal at Unity import. An importer that
already performs that conversion must not apply it a second time.

After import, apply the ghost's `mesh_alignment` as a child transform, then apply
the animated root. Its quaternion already includes the orientation estimate and
camera/frame conversions. Additional camera rotations or fixed Z-up corrections
would duplicate the conversion. The fixed 0.12 m display extent is illustrative,
not a measured physical object size. Pose estimation and mesh scale remain
approximate even when instruction offsets are exact.

Default playback spawns at 0 s, animates from 0.5–2.0 s, holds until 2.5 s, hides,
and restarts at 2.75 s. The current semantic specification does not configure these
times. Unity executes the supplied timeline rather than reinterpreting the task
text. See the [Unity guidance contract](docs/unity_guidance_contract.md) for the
complete transform, interpolation, pivot, and playback rules.

## Diagnostics

| Artifact | Purpose |
|---|---|
| `original.png`, `specification.json` | Capture and merged specification used by the run |
| `grounding.json`, `grounding.png` | Target/reference detections and annotated image |
| `segmentation_mask.png`, `preprocessing.json` | Full-scene SAM2 mask, box prompt, score, and crop bounds |
| `isolated.png` | Foreground with transparency and square padding |
| `reconstruction_input.png` | Exact gray-composited RGB input passed to TripoSR |
| `reconstructed.glb` | Centered raw reconstruction, before orientation alignment |
| `orientation.json` | Predicted relative angles, camera transforms, and alignment quaternion |
| `orientation_reference.png`, `orientation_target.png` | Actual paired-image inputs to Orient Anything V2 |
| `orientation_aligned.png` | Mesh rendered in the estimated target-camera orientation |
| `ghost.glb` | Embedded ghost asset, without the external alignment transform |
| `initial_ghost.glb`, `transformed_ghost.glb`, `demonstration.glb` | Ghosts with alignment/playback transforms applied in the illustrative preview layout |
| `preview.png`, `preview_anchors.json` | Guidance preview and its illustrative anchor positions |
| `guidance.json`, `timings.json` | Complete manifest and stage timings |
| `failure.json` | Error details and available timings when generation fails |

`preview.png` executes the same manifest and embedded mesh as Unity, but its anchor
positions are illustrative: the target is placed at the origin and reference
positions are approximated from image UV offsets using a fixed 0.6 m span. This
is not a measured reconstruction of the tabletop. The camera is displayed as a
point in a separate context panel. A placement arrow's exact length or direction
can differ from physical Unity playback because the two environments use different
anchor positions.

Opening `ghost.glb` or `reconstructed.glb` alone does not apply the manifest's
orientation correction. Inspect `preview.png`, the transformed diagnostic GLBs,
or a client that applies `mesh_alignment` when assessing the final guidance.

## Limitations

- Reconstruction and orientation are estimates; symmetry, occlusion, weak texture,
  and reconstruction distortions can make the initial pose inaccurate.
- Physical anchor localization is delegated to Unity. RGB-only metric depth,
  object dimensions, and surface boundaries are not recovered.
- Placement measures anchor-to-anchor offsets and does not enforce clearance,
  tabletop support, collision avoidance, or textual constraints.
- Spatial instructions use the captured camera heading. Estimated object pose
  does not establish an object-relative instruction frame.
- The current composer generates one target action per specification. It does not
  support articulated components, task planning, or arbitrary action sequences.
- No Unity application is included. The manifest is an execution contract for a
  separate client, and the HTTP bridge is intended for testing.

## Troubleshooting

| Symptom | Check |
|---|---|
| `right_of` produces leftward movement | It specifies a destination right of the reference anchor; use `direction: "right"` for a rightward displacement |
| Placement overlaps the laptop | Offsets are from its anchor, not its physical edge; clearance is unsupported |
| Forward follows an unexpected axis | Check captured heading and quaternion consistency; forward is camera-relative and horizontal |
| Ghost is horizontal despite a corrected preview | Apply `mesh_alignment.rotation_xyzw` to the imported mesh child before animation |
| Preview and Unity placement disagree | Preview reference positions are illustrative; inspect Unity's localized anchor positions and the returned numerical offsets |
| Grounding worker fails to initialize | Verify `.guidance-deps` and cached detector files; do not merge the conflicting Transformers stacks |
| `torchmcubes` import/build error | Apply the provided patch to the pinned TripoSR checkout |
| Hugging Face cache/model error at runtime | Download the listed files under the server account before starting offline inference |
| EGL context creation fails | Verify driver/OpenGL/EGL support with the installation smoke check |
| Server uses the wrong scenario | Inspect `/health`; restart with the intended `GUIDANCE_EXAMPLE` |
| Model memory is exhausted | Use one worker and check for other model/server processes |

## Tests

```sh
.venv310/bin/python -m unittest discover -s tests -v
```

Tests cover specification/frame validation, placement versus translation,
rotations and pivots, embedded assets, orientation conversion, headless rendering,
preprocessing, and the HTTP contract. Most use model substitutes; the renderer
test requires EGL. Real end-to-end verification requires a CLI run or server
request with installed checkpoints.

## Project layout

```text
guidance_composer/       Pipeline, model adapters, semantics, manifest, playback, preview
servers/                FastAPI bridge, test sender, orientation setup
examples/               Guidance specifications and test images
patches/                Patch for the pinned TripoSR source
requirements.txt        Main Python environment
requirements-grounding.txt  Isolated grounding compatibility stack
TripoSR/                External reconstruction source (ignored)
third_party/            External orientation source (ignored)
checkpoints/            Local SAM2 and orientation weights (ignored)
outputs/                Generated diagnostics (ignored)
docs/                   Unity contract and model-specific details
```

## Model references

[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM2](https://github.com/facebookresearch/sam2),
[TripoSR](https://github.com/VAST-AI-Research/TripoSR), and
[Orient Anything V2](https://github.com/SpatialVision/Orient-Anything-V2) are separate
upstream projects with their own source and model licenses.
