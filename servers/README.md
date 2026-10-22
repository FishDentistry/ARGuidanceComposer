# Guidance Composer testing server

A FastAPI bridge for Unity capture -> example semantics -> complete Unity manifest.
The server is under servers/ and uses the shared Grounding DINO -> SAM2 -> TripoSR -> Orient Anything V2 pipeline.

## Start

From the repository root, after [orientation setup](../docs/orientation.md)
(already completed in this workspace):

    .venv310/bin/python -m uvicorn servers.app:app --host 0.0.0.0 --port 8000 --workers 1

The models initialize once at startup. Wait for "Application startup complete"
before sending captures. The default semantic specification is
examples/pop_can_test.json. Select another example before starting:

    GUIDANCE_EXAMPLE=apple_right_of.json \
      .venv310/bin/python -m uvicorn servers.app:app --host 0.0.0.0 --port 8000 --workers 1

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| GUIDANCE_EXAMPLE | pop_can_test.json | JSON filename inside this repository's examples/ directory |
| GUIDANCE_DEVICE | cuda:0 | Model device; orientation rendering additionally requires headless EGL support |
| GUIDANCE_SAM2_CHECKPOINT | checkpoints/sam2.1_hiera_small.pt (absolute repository path) | Local SAM2 weights; relative overrides resolve from the working directory |
| GUIDANCE_SAM2_CONFIG | configs/sam2.1/sam2.1_hiera_s.yaml | Matching configuration from the installed SAM2 package |
| GUIDANCE_ORIENTATION_CHECKPOINT | checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt (absolute repository path) | Local V2 weights |
| GUIDANCE_ORIENTATION_SOURCE | third_party/orient_anything_v2 (absolute repository path) | Official V2 inference source |

Use one Uvicorn worker so there is one model copy on the GPU. Avoid --reload when
testing model reuse: reload restarts the process and loads the models again.
A dedicated one-thread model executor loads models, processes captures
sequentially, and closes the grounding subprocess on shutdown. Health requests
remain responsive while inference runs.

For a Unity device on another machine, use this computer's reachable LAN address,
e.g. http://192.168.1.20:8000/guidance. 0.0.0.0 is a bind address, not a client URL.
This is an unauthenticated development/testing server.

## Request from Unity

POST /guidance with Content-Type: application/json:

    {
      "scene_image_base64": "<Base64 PNG or JPEG file bytes>",
      "camera_pose": {
        "position": [0.0, 1.6, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0]
      }
    }

Use raw Base64 (e.g. Convert.ToBase64String over encoded PNG/JPEG bytes), without
a data:image/... prefix. Position is in meters; rotation is a unit quaternion
[x,y,z,w]. Capture image and pose at the same moment, in Unity's Y-up/+Z-forward
world convention. Vector/quaternion fields are JSON arrays, not objects with
x/y/z property names.

Unity need not send target descriptions, actions, constraints, context, mesh data,
object IDs, or paths. The server:

1. Loads the selected example's semantic fields on every request.
2. Decodes the supplied image into memory.
3. Replaces the example's image and camera pose with the request capture.
4. Derives gravity-aligned camera_axes_world from that captured pose.
5. Grounds the target with Grounding DINO, segments it with a SAM2 box prompt,
   runs TripoSR, estimates mesh-to-capture alignment with Orient Anything V2,
   and composes guidance.
6. Saves the usual diagnostic artifacts in a new directory under outputs/.
7. Returns the complete manifest directly as application/json.

Example camera_pose and camera_axes_world values are never reused for a new
capture. Edits to the selected example's action/target/constraints/context apply
to the next request without reloading models. To change which example is selected,
restart with a different GUIDANCE_EXAMPLE value.

An optional camera_axes_world object can accompany the capture:

    "camera_axes_world": {
      "right": [1, 0, 0],
      "up": [0, 1, 0],
      "forward": [0, 0, 1]
    }

Supplied axes are validated against the pose and gravity convention. They are
useful when a near-vertical view has no reliable heading derivable from its
forward vector. Without reliable supplied axes, that pose returns HTTP 422;
the server never borrows the example's heading.

PNG/JPEG and the first frame of JPEG-compatible MPO photos are accepted.
The maximum compressed image-file size is 24 MiB (approximately 32 MiB Base64),
and maximum decoded size is 40 megapixels. JPEG EXIF orientation is applied
before composition; returned source dimensions/UV anchors refer to the oriented
image. Unity-encoded PNG/JPEG captures normally have no EXIF rotation.

## Response

HTTP 200 is the guidance manifest itself, with no additional wrapper:

    {
      "schema_version": "1.0",
      "coordinate_frame": { "...": "frozen captured axes" },
      "capture": { "...": "capture pose and image metadata" },
      "anchors": [ "..." ],
      "assets": [ { "format": "glb", "encoding": "base64", "data": "..." } ],
      "visual_objects": [ "..." ],
      "timeline": [ "..." ],
      "playback": { "..." : "..." },
      "limitations": { "..." : "..." },
      "semantic_context": { "..." : "..." },
      "timings_ms": { "..." : "..." }
    }

The above is schematic; actual responses contain all numerical values and embedded
GLB bytes. Unity can execute the response using the existing
[Unity guidance contract](../docs/unity_guidance_contract.md). No generated
server-side asset paths or directories are needed by Unity. Each valid capture
also calls composer.generate(spec, output=...) to save the same artifacts as the
example runner under this repository's outputs/ directory, independent of the
server's working directory. For example:

    outputs/server_pop_can_test_20261022T180000_123456Z_a1b2c3d4/

Names include the selected example, a UTC timestamp, and a random suffix so
requests have separate directories. The server logs the directory path. Each
successful run includes original.png, specification.json, grounding.json,
grounding.png, isolated.png, reconstruction_input.png, segmentation_mask.png,
preprocessing.json, reconstructed.glb,
ghost.glb, initial_ghost.glb, transformed_ghost.glb, demonstration.glb,
preview_anchors.json, preview.png, guidance.json, and timings.json. Orient Anything V2
also writes orientation.json, orientation_reference.png, orientation_target.png,
and orientation_aligned.png.
The saved guidance.json matches the complete HTTP response. Preview anchors
retain the example runner's illustrative convention, with the target at the
origin and the camera pose displayed.

segmentation_mask.png is SAM2's full-scene mask. preprocessing.json records the
box prompt, predicted mask quality, foreground pixel count, and crop bounds.
See [SAM2 setup](../docs/segmentation.md); restart the server after upgrading from
the earlier rembg preprocessing implementation or adding orientation estimation.
The response carries the static correction in ghost_target.mesh_alignment.rotation_xyzw;
Unity must apply that local mesh-child rotation before the animated root. There
are no new Unity request fields. The model loads once and is reused across captures.

If generation fails after composition starts, the directory retains available
intermediate artifacts and failure.json. Invalid requests rejected before
composition do not create output directories. Diagnostic export runs before the
response is returned; its time is included in timings_ms.

The selected example may require another object (e.g. a laptop for right_of);
that reference is grounded automatically as usual. Placement offsets remain
exact meters relative to Unity-localized image anchors. Ghost shape/scale/alignment
remain approximate, with the existing explicit flags. Constraint/collision/surface
limitations remain unchanged.

Set the Unity request timeout to allow model inference and queued requests;
180 seconds is a reasonable testing value. This endpoint returns the completed
result synchronously rather than a job ID or streaming response.

## Health, API docs, and errors

- GET /health returns ready status, selected example, device, and serialized
  request behavior after model startup.
- GET /docs provides FastAPI's interactive request/response API documentation.
- GET /openapi.json provides the request schema.

Errors use FastAPI's detail field:

| Status | Meaning |
|---|---|
| 422 | Invalid request/image/pose/axes/combined spec, or no matching grounded target/reference |
| 409 | Ambiguous grounding; detail.candidates contains competing detections |
| 500 | Generation failed; check server logs |
| 503 | Server not ready or configured example unavailable |

For example:

    {
      "detail": {
        "code": "ambiguous_grounding",
        "message": "Ambiguous grounding: multiple plausible regions. Refine the description.",
        "candidates": [ "..." ]
      }
    }

Malformed schema requests return detail.code invalid_request with field locations
and messages. Large Base64 payloads are not echoed back in validation errors.

## Send a test capture without Unity

With the server running:

    .venv310/bin/python -m servers.send_test \
      --image examples/images/apple_test.jpg \
      --camera-spec examples/apple_forward_new_cam.json \
      --output /tmp/unity_guidance_response.json

The sender reads only camera_pose from --camera-spec and transmits it with the
image. The semantic action still comes from the server's GUIDANCE_EXAMPLE.
--include-axes also transmits that file's captured axes. Use --url to target a
different machine/port, and --timeout to change the 180-second timeout.

The sender uses Python's standard HTTP library and prints a small response summary;
the optional --output saves the entire returned manifest. This file is a client
testing convenience, not a requirement for Unity.

## Dependencies and tests

Server and HTTP test dependencies are included in the root requirements.txt.
Use the [root installation guide](../README.md#installation) to install the shared
environment, external model source, and checkpoints. requirements-grounding.txt
remains separate because its Transformers versions are isolated from TripoSR.
The JSON protocol does not require python-multipart.

    .venv310/bin/python -m unittest discover -s tests -p test_server.py -v
    .venv310/bin/python -m unittest discover -s tests -v

The HTTP tests use lightweight model stand-ins with the real composer to verify
manifest packaging without reloading GPU models. They cover camera/template
merging, axes derivation, template edits, image formats, error responses,
serialized concurrent inference, responsive health checks, model shutdown,
per-request diagnostic artifacts, and saved failure details.

The implementation follows FastAPI's
[lifespan startup/shutdown pattern](https://fastapi.tiangolo.com/advanced/events/).

Validated on October 22, 2026: all 15 server tests and all 43 repository tests
pass. A real localhost HTTP request using apple_test.jpg and the +90-degree
camera pose returned HTTP 200 through Grounding DINO and TripoSR. The complete
response passed manifest validation and GLB decoding, and its placement endpoint
used the incoming rotated axes correctly. The first request took about 28.3 s
after startup, including grounding/reconstruction; this is one measurement.
