# Renderforge — Inference Service

Standalone FastAPI microservice wrapping **TripoSR** for image → 3D mesh
generation. This is the "GPU Worker / AI 3D Generation Pipeline" component in
the Renderforge architecture — it does not do auth, billing, job queueing, or
permanent storage. Your Django + Celery layer calls this service over HTTP
for the actual GPU compute, then uploads the returned mesh to R2/S3 itself.

```
Celery worker --POST /infer--> [this service] --GLB/OBJ file--> Celery worker --> R2/S3
```

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate

# Install torch matching your CUDA driver (check nvidia-smi)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# TripoSR isn't on PyPI — install from source
git clone https://github.com/VAST-AI-Research/TripoSR
pip install -e TripoSR

cp .env.example .env   # then edit RF_API_KEY etc.
```

The first request (or server startup) will download TripoSR's weights from
Hugging Face (`stabilityai/TripoSR`, ~1.5GB) and cache them locally.

## 2. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

GPU memory: TripoSR comfortably fits on 6GB VRAM at the default chunk size.
If you hit OOM on a smaller card, lower `RF_MODEL_CHUNK_SIZE` in `.env`
(e.g. `4096` or `2048` — slower but less memory).

## 3. API

### `GET /health`
Returns model load status and GPU stats. Poll this before routing traffic.

### `POST /infer`
`multipart/form-data`:

| field               | type | default  | notes                          |
|---------------------|------|----------|---------------------------------|
| `image`             | file | required | png/jpg/webp, ≤10MB              |
| `quality`           | str  | `medium` | `low` (128) / `medium` (256) / `high` (320) marching-cubes resolution |
| `output_format`     | str  | `glb`    | `glb` or `obj`                  |
| `remove_background` | bool | `true`   | runs rembg before inference     |
| `foreground_ratio`  | float| `0.85`   | how tightly to crop the subject |

```bash
curl -X POST http://localhost:8001/infer \
  -H "X-API-Key: change-me" \
  -F "image=@chair.jpg" \
  -F "quality=high" \
  -F "output_format=glb"
```

Response:
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "completed",
  "output_format": "glb",
  "mesh_url": "/download/a1b2c3d4e5f6",
  "vertex_count": 8421,
  "face_count": 16840,
  "inference_seconds": 4.31,
  "device": "cuda"
}
```

### `GET /download/{job_id}`
Streams the generated mesh file.

### `DELETE /jobs/{job_id}`
Deletes local scratch files. Call this from the Celery worker right after it
has copied the mesh to permanent storage (R2/S3), since this service does
**not** clean up on its own — see `RF_JOB_TTL_SECONDS` for a fallback if you
add a cron/sweeper later.

## 4. Running on your local GPU (no Docker)

This is the recommended path for a single dev machine with one consumer GPU
(e.g. GTX 1650/1660/3060) rather than the Docker/cloud route below.

```bash
# 1. Confirm torch sees your card and get a suggested chunk size
python scripts/check_gpu.py

# 2. Put that suggested value in .env
echo "RF_MODEL_CHUNK_SIZE=4096" >> .env   # example — use what the script suggests
echo "RF_DEVICE=cuda" >> .env

# 3. Run directly on the host — no container needed
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Notes specific to running on a local/consumer GPU:

- **VRAM is the real constraint**, not compute. If `/infer` throws a CUDA
  OOM error, lower `RF_MODEL_CHUNK_SIZE` first (try 4096, then 2048) before
  touching `quality` — chunk size controls renderer memory, `quality`
  controls mesh resolution.
- **Only run one inference at a time** on a single consumer GPU. FastAPI
  will happily accept concurrent requests, but the model isn't safe for
  parallel forward passes on one device — put `--workers 1` on uvicorn and
  let Celery's queue serialize jobs rather than sending them in parallel.
- **Close other GPU-heavy apps** (games, other model servers) before running
  generation jobs — TripoSR + rembg need roughly 3-5GB free VRAM at default
  settings.
- **First request is slow.** Model weights download + load onto GPU take
  10-60s; this happens once at server startup (via `lifespan`), not on the
  first `/infer` call, so don't mistake startup time for inference time.

## 5. Choosing the model (engine)

One engine is loaded per process, set with `RF_ENGINE` in `.env`. Two do not
fit together in 4-6 GB of VRAM, so switching means restarting the server.

| `RF_ENGINE` | Model | Output | Pick it when |
|---|---|---|---|
| `triposr` (default) | TripoSR | Vertex-colored mesh | You want color and the fastest result |
| `hunyuan3d_mini` | Hunyuan3D-2mini (shape model) | **Untextured grey mesh**, no colors | You want cleaner geometry and can live without color |

`quality` maps to engine settings:

| quality | TripoSR (marching cubes) | Hunyuan3D-2mini (steps, octree) |
|---|---|---|
| low | 128 | 20, 256 |
| medium | 256 | 30, 320 |
| high | 320 | 50, 380 |

Install for Hunyuan3D-2mini:
```bash
git clone https://github.com/Tencent/Hunyuan3D-2
pip install -r Hunyuan3D-2/requirements.txt
pip install -e Hunyuan3D-2
echo "RF_ENGINE=hunyuan3d_mini" >> .env
```
`GET /health` and every `/infer` response report which engine is running.

Notes:
- Hunyuan3D-2mini here is **shape only**. Hunyuan's texture model needs compiled
  CUDA extensions and much more VRAM, so it is not wired in.
- A third-party Windows guide lists 4 GB as the minimum and 6 GB or more as
  recommended for Hunyuan3D shape-only generation (it covers Hunyuan3D-2, not
  specifically the mini model). Treat that as a guide, and if you hit
  out-of-memory, lower `RF_HUNYUAN_NUM_CHUNKS`.
- **Not yet run on a real GPU.** The engine wiring is tested with stand-in
  modules (right arguments passed per quality level, mesh export, `/health`),
  but the real model calls are not.

## 6. Pipeline stages (matches the architecture diagram)

| Diagram stage         | Implementation                                              |
|------------------------|--------------------------------------------------------------|
| Preprocess Image       | `rembg` background removal + recenter (`app/inference.py`)  |
| Image to 3D (AI Model) | TripoSR forward pass + marching cubes extraction            |
| Mesh Processing        | `trimesh` cleanup: merge verts, drop duplicate faces, fix normals |
| Texture Generation     | Provided by TripoSR's vertex-colored output (no separate PBR baking pass yet — see below) |
| UV Unwrapping          | Not implemented — TripoSR outputs vertex colors, not a UV-mapped texture. Add `xatlas` + a texture-baking step here if you need UV'd textures for game engines. |
| Optimization           | `trimesh.simplify_quadric_decimation` caps output to `RF_DECIMATION_TARGET_FACES` |

## 7. Known limitations / next steps

- **Single-image only.** TripoSR is a single-view reconstruction model —
  accuracy on complex/occluded geometry will be lower than multi-view
  approaches (Wonder3D, Zero123++). Swapping the model is isolated to
  `app/model.py` and `app/inference.py`.
- **No thumbnail rendering yet.** `thumbnail_url` is always `null`; wire up
  `pyrender`/`trimesh.Scene.save_image` in `run_inference()` if you want a
  preview image without a full download.
- **Synchronous inference.** One request blocks the GPU until it finishes
  (~3–8s typically). Run one process per GPU and let Celery's queue handle
  concurrency across workers rather than adding internal queueing here.
- **No auth beyond a static API key.** Fine behind an internal network /
  Cloudflare Tunnel restricted to your Django backend; don't expose this
  service directly to the internet.

## 8. Docker (alternative to local GPU run)

```bash
docker build -t renderforge-inference .
docker run --gpus all -p 8001:8001 --env-file .env renderforge-inference
```
