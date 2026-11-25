# RenderForge 🚀
### Open-Source AI-Powered 3D Model Generation Backend

RenderForge is an open-source, AI-powered 3D model generation platform that converts ordinary 2D images into 3D models using deep learning.

The project combines **FastAPI, Python, PyTorch, TripoSR, Hunyuan3D-2mini, Trimesh and PostgreSQL** to provide an API-driven 3D generation workflow.

The goal is to make AI-based 3D generation accessible to developers, students, researchers and independent creators using local GPU hardware or compatible cloud GPU environments.

> **Project status:** Active development. The backend and inference service are implemented, but some production features and integrations are still under development.

---

## 📌 Table of Contents

- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Technology Stack](#-technology-stack)
- [How RenderForge Works](#-how-renderforge-works)
- [AI Inference Pipeline](#-ai-inference-pipeline)
- [Generation Logic](#-generation-logic)
- [Database Design](#-database-design)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Environment Configuration](#-environment-configuration)
- [Running the Project](#-running-the-project)
- [GPU Configuration](#-gpu-configuration)
- [Testing](#-testing)
- [Current Limitations](#-current-limitations)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 1. AI-powered image-to-3D generation

- Convert a single 2D image into a 3D mesh.
- Use TripoSR as the default image-to-3D engine.
- Optional Hunyuan3D-2mini shape-generation engine.
- Support PNG, JPG and WEBP input images.
- Generate GLB and OBJ output formats.
- Configurable generation quality.

### 2. AI inference service

- Separate FastAPI inference microservice.
- Load AI model weights once during startup.
- Automatic GPU/CPU device selection.
- GPU information and VRAM monitoring.
- Configurable model chunk size.
- API-key-based communication between backend and inference service.

### 3. Mesh processing

- Remove invalid mesh values.
- Merge duplicate vertices.
- Remove duplicate faces.
- Remove unreferenced vertices.
- Repair mesh normals.
- Optional polygon decimation.
- Export optimized mesh files.

### 4. User authentication

- User registration.
- User login.
- Password hashing.
- JWT access tokens.
- JWT refresh tokens.
- Current-user endpoint.
- User-specific generation history.

### 5. Generation management

- Asynchronous HTTP generation workflow.
- Background generation tasks.
- Generation status tracking.
- Generation history.
- Mesh download endpoint.
- Generation error reporting.
- GPU concurrency control.

### 6. Database integration

- PostgreSQL support.
- Neon PostgreSQL support.
- SQLAlchemy asynchronous database access.
- Alembic database migrations.
- SQLite-compatible database utilities for testing.

---

## 🏗️ Project Architecture

RenderForge uses a two-service architecture.

```mermaid
flowchart TD
    A["Client / API Consumer"] --> B["FastAPI Backend :8000"]
    B --> C["JWT Authentication"]
    C --> D["PostgreSQL / Neon"]
    B --> E["Generation Service"]
    E --> F["Local Input Storage"]
    E --> G["Background Task"]
    G --> H["Inference HTTP Client"]
    H --> I["AI Inference Service :8001"]
    I --> J["TripoSR / Hunyuan3D"]
    J --> K["Mesh Processing"]
    K --> L["GLB / OBJ Export"]
    L --> I
    I --> H
    H --> M["Local Mesh Storage"]
    M --> N["Download API"]
    N --> A
```

### Service responsibilities

| Service | Responsibility |
|---|---|
| Backend API | Authentication, user management and generation requests |
| Database | Users, generation records and generation metadata |
| Generation service | Job lifecycle and background processing |
| Inference service | AI model loading and image-to-3D generation |
| GPU | AI model execution |
| Trimesh | Mesh cleanup, optimization and export |
| Local storage | Uploaded images and generated meshes |

**Important:** The inference service does not manage user authentication or permanent storage. The main backend handles those responsibilities.

---

## 🧰 Technology Stack

| Category | Technology |
|---|---|
| Backend framework | FastAPI |
| Programming language | Python |
| AI framework | PyTorch |
| Default AI model | TripoSR |
| Optional AI model | Hunyuan3D-2mini |
| Image processing | Pillow, rembg |
| Mesh processing | Trimesh |
| Mesh simplification | fast-simplification |
| Database | PostgreSQL |
| Cloud database option | Neon |
| ORM | SQLAlchemy |
| Database migrations | Alembic |
| Authentication | JWT |
| Password security | bcrypt |
| HTTP communication | HTTPX |
| API server | Uvicorn |
| Testing | Pytest-compatible test suite |

---

## ⚙️ How RenderForge Works

The generation workflow consists of the following stages.

### Stage 1: User authentication

A user creates an account or logs in.

The backend verifies the credentials and generates:

- Access token
- Refresh token

The access token is used to authenticate protected API requests.

### Stage 2: Image upload

The authenticated user submits an image to the generation endpoint.

Supported image formats:

| Format | MIME type |
|---|---|
| PNG | image/png |
| JPG / JPEG | image/jpeg |
| WEBP | image/webp |

The backend validates the uploaded image, checks the format and applies the configured upload-size limit.

### Stage 3: Generation record creation

The backend creates a generation record in PostgreSQL.

Initial generation status:

`PENDING`

The original image is stored locally, and the generation service schedules the processing task using FastAPI BackgroundTasks.

The API responds with HTTP `202 Accepted`.

### Stage 4: AI inference

The background generation service sends the image to the separate inference service.

The inference service:

1. Receives the uploaded image.
2. Validates the image.
3. Selects the configured AI engine.
4. Applies image preprocessing.
5. Runs the AI model.
6. Extracts a 3D mesh.
7. Returns generation metadata.

### Stage 5: Mesh processing

The generated mesh is passed through the shared processing pipeline.

Processing includes:

- Mesh cleanup.
- Vertex merging.
- Duplicate-face removal.
- Normal correction.
- Optional polygon decimation.

### Stage 6: Mesh export

The processed mesh is exported into the requested format:

- GLB
- OBJ

The inference service temporarily stores the exported mesh and provides a download endpoint.

### Stage 7: Backend storage

The backend downloads the generated mesh from the inference service and saves it in its local storage directory.

The generation record is updated with:

- Generation status.
- Output file location.
- Vertex count.
- Face count.
- Completion timestamp.

Final status:

`COMPLETED`

### Stage 8: Download

The authenticated user can download the generated 3D model through the backend download endpoint.

---

## 🧠 AI Inference Pipeline

RenderForge separates the AI model implementation from the main backend.

This allows developers to add or replace inference engines without rewriting authentication or generation-management logic.

### Supported engines

#### TripoSR

<details>
<summary>TripoSR — Default engine</summary>

TripoSR is a fast single-image-to-3D reconstruction model.

Repository: https://github.com/VAST-AI-Research/TripoSR

Model weights: https://huggingface.co/stabilityai/TripoSR

Features:

- Single-image 3D reconstruction.
- Vertex-colored mesh output.
- Marching-cubes mesh extraction.
- Configurable renderer chunk size.

</details>

#### Hunyuan3D-2mini

<details>
<summary>Hunyuan3D-2mini — Optional engine</summary>

Hunyuan3D-2mini is an alternative shape-generation engine.

Repository: https://github.com/Tencent/Hunyuan3D-2

Model weights: https://huggingface.co/tencent/Hunyuan3D-2mini

Features:

- Shape generation.
- Configurable inference steps.
- Configurable octree resolution.
- Trimesh output.

**Current implementation:** Shape generation only. Separate texture generation and PBR texture baking are not implemented.

</details>

### Engine selection

Configure the inference engine using:

```env
RF_ENGINE=triposr
```

Or:

```env
RF_ENGINE=hunyuan3d_mini
```

Only one engine is loaded per inference process because loading multiple large models simultaneously can exceed available GPU memory.

### Quality settings

| Quality | TripoSR resolution | Hunyuan steps | Hunyuan octree |
|---|---:|---:|---:|
| Low | 128 | 20 | 256 |
| Medium | 256 | 30 | 320 |
| High | 320 | 50 | 380 |

Higher quality can increase processing time and memory usage.

---

## 🔄 Generation Logic

The generation lifecycle uses four states:

```mermaid
stateDiagram-v2
    [*] --> PENDING
    PENDING --> PROCESSING
    PROCESSING --> COMPLETED
    PROCESSING --> FAILED
    COMPLETED --> [*]
    FAILED --> [*]
```

### Generation status explanation

| Status | Description |
|---|---|
| `pending` | Request accepted and waiting for processing |
| `processing` | AI inference and mesh processing are running |
| `completed` | Mesh generated and stored successfully |
| `failed` | Generation failed |

### Generation service logic

The main implementation is located at:

`app/services/generations.py`

Simplified workflow:

```python
async def process_generation(generation_id):

    # 1. Retrieve generation
    generation = await get_generation(generation_id)

    # 2. Update status
    generation.status = "processing"

    # 3. Read uploaded image
    image = storage.read(generation.input_key)

    # 4. Call AI inference service
    async with gpu_semaphore:
        result = await inference.generate_mesh(image)

    # 5. Save generated mesh
    storage.write(result.mesh)

    # 6. Update metadata
    generation.vertex_count = result.vertex_count
    generation.face_count = result.face_count

    # 7. Complete generation
    generation.status = "completed"
```

This is explanatory pseudocode. The actual implementation also handles database sessions, exceptions, timestamps and cleanup.

### GPU concurrency logic

The generation service uses an `asyncio.Semaphore` to limit the number of simultaneous inference requests.

Default configuration:

```env
RF_MAX_CONCURRENT_GENERATIONS=1
```

For a single consumer GPU, one generation at a time is the recommended starting point.

---

## 🗄️ Database Design

RenderForge currently uses two main application tables.

### Users table

Stores account information.

| Column | Description |
|---|---|
| id | Unique user UUID |
| email | User email address |
| hashed_password | Hashed password |
| full_name | Optional user name |
| is_active | Account status |
| created_at | Account creation timestamp |

### Generations table

Stores 3D generation metadata.

| Column | Description |
|---|---|
| id | Unique generation UUID |
| user_id | Associated user |
| status | Generation status |
| quality | Requested generation quality |
| output_format | GLB or OBJ |
| source_filename | Original image filename |
| input_key | Input storage key |
| output_key | Output storage key |
| vertex_count | Generated mesh vertex count |
| face_count | Generated mesh face count |
| error_message | Failure information |
| created_at | Generation creation time |
| updated_at | Last update time |
| completed_at | Completion time |

### Database relationship

```mermaid
erDiagram
    USERS ||--o{ GENERATIONS : creates

    USERS {
        uuid id PK
        string email UK
        string hashed_password
        string full_name
        boolean is_active
        datetime created_at
    }

    GENERATIONS {
        uuid id PK
        uuid user_id FK
        string status
        string quality
        string output_format
        string source_filename
        string input_key
        string output_key
        integer vertex_count
        integer face_count
        text error_message
        datetime created_at
        datetime completed_at
    }
```

### Neon PostgreSQL

Neon can be used as the cloud-hosted PostgreSQL database.

Only generation metadata and user records are stored in Neon in the current implementation.

**Generated mesh files remain in local storage.** They are not automatically synchronized to Neon or a cloud object-storage provider.

See [NEON.md](NEON.md) for database setup instructions.

---

## 🌐 API Documentation

When the backend is running, open:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Authentication endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/signup` | Register a user |
| POST | `/auth/login` | Authenticate a user |
| POST | `/auth/refresh` | Exchange refresh token for a new token pair |
| GET | `/auth/me` | Get authenticated user information |

#### Signup request

```json
{
  "email": "user@example.com",
  "password": "StrongPassword123",
  "full_name": "RenderForge User"
}
```

#### Login request

The login endpoint uses OAuth2 password form data:

```text
username=user@example.com
password=StrongPassword123
```

#### Token response

```json
{
  "access_token": "ACCESS_TOKEN",
  "refresh_token": "REFRESH_TOKEN",
  "token_type": "bearer"
}
```

### Generation endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/generations` | Submit an image for 3D generation |
| GET | `/generations` | List the authenticated user's generations |
| GET | `/generations/{generation_id}` | Get generation details |
| GET | `/generations/{generation_id}/download` | Download generated mesh |

#### Create generation request

Endpoint:

```text
POST /generations
```

Headers:

```http
Authorization: Bearer ACCESS_TOKEN
```

Content type:

```text
multipart/form-data
```

Fields:

| Field | Type | Default |
|---|---|---|
| image | File | Required |
| quality | String | medium |
| output_format | String | glb |

Example:

```bash
curl -X POST http://localhost:8000/generations \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "image=@chair.png" \
  -F "quality=medium" \
  -F "output_format=glb"
```

#### Generation response

The endpoint returns HTTP `202 Accepted`.

```json
{
  "id": "GENERATED-UUID",
  "status": "pending",
  "quality": "medium",
  "output_format": "glb",
  "source_filename": "chair.png",
  "vertex_count": null,
  "face_count": null,
  "error_message": null,
  "download_url": null
}
```

Poll the generation endpoint to check its status.

### Inference service endpoints

The inference service runs separately on port `8001`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Model and GPU health |
| POST | `/infer` | Generate a 3D mesh |
| GET | `/download/{job_id}` | Download temporary mesh |
| DELETE | `/jobs/{job_id}` | Delete temporary inference output |

The `/infer` endpoint accepts multipart form data and uses the configured API key.

Example:

```bash
curl -X POST http://localhost:8001/infer \
  -H "X-API-Key: YOUR_INFERENCE_API_KEY" \
  -F "image=@chair.png" \
  -F "quality=medium" \
  -F "output_format=glb"
```

### Health checks

Backend:

```text
GET /health
GET /health/db
```

Inference:

```text
GET /health
```

---

## 📁 Project Structure

```text
renderforgebackend/
│
├── app/
│   ├── main.py
│   ├── factory.py
│   ├── config.py
│   ├── db.py
│   ├── deps.py
│   ├── images.py
│   ├── models.py
│   ├── schemas.py
│   ├── security.py
│   │
│   ├── routers/
│   │   ├── auth.py
│   │   └── generations.py
│   │
│   └── services/
│       ├── generations.py
│       ├── inference_client.py
│       └── storage.py
│
├── inference/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── engines.py
│   │   └── inference.py
│   │
│   ├── tests/
│   ├── requirements.txt
│   ├── .env.example
│   ├── Dockerfile
│   └── README.md
│
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 0001_users_and_generations.py
│
├── tests/
│   ├── test_auth.py
│   ├── test_generations.py
│   └── test_database.py
│
├── storage/
├── .github/
├── .env.example
├── .gitignore
├── alembic.ini
├── requirements.txt
├── NEON.md
├── CONTRIBUTING.md
└── LICENSE
```

Some folders, such as `storage/` and `.github/`, may be created locally or as the project develops.

---

## 💻 Installation

### Prerequisites

Install the following:

- Python 3.11 recommended for the inference environment.
- Git.
- NVIDIA GPU with a compatible CUDA-enabled PyTorch installation, or CPU for supported workloads.
- PostgreSQL or a Neon PostgreSQL database.
- NVIDIA drivers if using an NVIDIA GPU.

Clone the repository:

```bash
git clone https://github.com/vigneshuvn-21ITL133/renderforgebackend.git

cd renderforgebackend
```

### 1. Setup the backend

Create a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

**Linux / WSL**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure backend environment

Copy the example environment file:

**Windows**

```powershell
Copy-Item .env.example .env
```

**Linux / WSL**

```bash
cp .env.example .env
```

Edit `.env` with your database connection and secrets.

### 3. Setup the inference service

Open another terminal:

```bash
cd inference
```

Create a separate environment:

```bash
python -m venv .venv
```

Activate it.

Install a PyTorch build compatible with your NVIDIA driver from:

https://pytorch.org/get-started/locally/

For example, a CUDA 12.1 installation:

```bash
pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the inference requirements:

```bash
pip install -r requirements.txt
```

Install TripoSR from its source repository:

```bash
git clone https://github.com/VAST-AI-Research/TripoSR

pip install -e TripoSR
```

Copy the environment configuration:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The first model startup downloads the required model weights. Ensure that you have enough disk space and a working internet connection.

For Hunyuan3D-2mini, install the optional upstream package and its dependencies as described in [inference/README.md](inference/README.md).

---

## 🔐 Environment Configuration

### Backend `.env`

```env
RF_DATABASE_URL=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require

RF_CREATE_TABLES_ON_STARTUP=false

RF_JWT_SECRET_KEY=YOUR_RANDOM_SECRET_AT_LEAST_32_CHARACTERS

RF_INFERENCE_BASE_URL=http://localhost:8001

RF_INFERENCE_API_KEY=YOUR_SHARED_INFERENCE_API_KEY

RF_ALLOWED_ORIGINS=["http://localhost:5173"]
```

Notes:

- Use your own PostgreSQL or Neon connection string.
- Run migrations before setting `RF_CREATE_TABLES_ON_STARTUP=false`.
- Use a secure random JWT secret.
- The inference API key must match the key configured in the inference service.
- Change the CORS origin to match your actual frontend origin.

### Inference `.env`

```env
RF_HOST=0.0.0.0
RF_PORT=8001

RF_API_KEY=YOUR_SHARED_INFERENCE_API_KEY

RF_ENGINE=triposr

RF_DEVICE=auto

RF_MODEL_CHUNK_SIZE=2048

RF_MAX_UPLOAD_MB=10

RF_ENABLE_MESH_CLEANUP=true

RF_ENABLE_DECIMATION=true

RF_DECIMATION_TARGET_FACES=20000
```

**Security:** Do not leave `RF_API_KEY` unset when exposing the inference service. The current inference implementation disables API-key enforcement when the variable is empty.

Never commit `.env` files, credentials or private model files to GitHub.

---

## ▶️ Running the Project

Run the services in separate terminals.

### Terminal 1: Start the database

Start your local PostgreSQL instance or configure Neon.

Apply database migrations:

```bash
alembic upgrade head
```

### Terminal 2: Start the inference service

```bash
cd inference

# Activate the inference virtual environment first

uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The service loads the selected AI model during startup.

Check:

```text
http://localhost:8001/health
```

### Terminal 3: Start the backend

From the repository root:

```bash
# Activate the backend virtual environment first

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check:

```text
http://localhost:8000/health
```

Database health:

```text
http://localhost:8000/health/db
```

API documentation:

```text
http://localhost:8000/docs
```

---

## 🎮 GPU Configuration

RenderForge supports automatic device selection.

```env
RF_DEVICE=auto
```

Available options:

| Value | Behaviour |
|---|---|
| `auto` | Select CUDA if available, otherwise CPU |
| `cuda` | Use NVIDIA GPU |
| `cpu` | Use CPU |

Check GPU availability:

```bash
nvidia-smi
```

Check PyTorch CUDA support:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory)
```

### Recommended settings for smaller GPUs

For a GPU with 4GB VRAM, start with:

```env
RF_ENGINE=triposr
RF_DEVICE=auto
RF_MODEL_CHUNK_SIZE=2048
RF_MAX_UPLOAD_MB=5
RF_ENABLE_MESH_CLEANUP=true
RF_ENABLE_DECIMATION=true
RF_DECIMATION_TARGET_FACES=15000
```

These are experimental starting values, not guaranteed to fit every image or model configuration.

Tips:

- Generate one model at a time.
- Start with low quality.
- Use small input images.
- Reduce model chunk size if CUDA runs out of memory.
- Close other GPU-heavy applications.
- Use cloud GPU notebooks for heavier experiments when necessary.

**Hardware note:** TripoSR's documented default configuration is designed around approximately 6GB of VRAM. A GTX 1650 with 4GB is below that reference, so low-memory settings may still encounter out-of-memory errors.

Free GPU experimentation resources:

- [Kaggle Notebooks](https://www.kaggle.com/code)
- [Google Colab](https://colab.research.google.com/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🧪 Testing

The repository includes tests for backend authentication, generation workflows and database utilities, along with inference-service tests.

Install pytest if it is not already available:

```bash
pip install pytest
```

Run the backend tests:

```bash
python -m pytest tests/ -v
```

Run inference tests:

```bash
cd inference

python -m pytest tests/ -v
```

The presence of tests does not mean every real GPU model has been validated. Real TripoSR and Hunyuan3D inference should be tested on compatible hardware separately.

---

## ⚠️ Current Limitations

RenderForge is under active development. The following limitations are important:

1. **Single-image reconstruction:** Current generation is based on a single input image. Hidden geometry and complex objects may not be reconstructed accurately.
2. **GPU requirements:** Larger models and higher-quality settings require more VRAM.
3. **Local file storage:** Uploaded images and generated meshes are stored locally. They are not automatically available across different backend machines.
4. **Background tasks:** The backend currently uses FastAPI BackgroundTasks rather than a durable external job queue. Jobs can be interrupted by process restarts.
5. **Restart recovery:** Pending and processing jobs are marked failed when the backend starts. This recovery approach is intended for a single-instance setup, not multiple concurrent backend replicas.
6. **Hunyuan texture generation:** The current Hunyuan3D-2mini integration generates shape geometry only.
7. **UV mapping:** A complete UV unwrapping and texture-baking workflow is not implemented.
8. **Thumbnail generation:** The inference response currently returns no generated thumbnail.
9. **Production deployment:** Additional security, persistent object storage, job orchestration and operational monitoring are required before production use.

---

## 🛣️ Roadmap

The following are planned improvements, not claims about existing functionality.

### Phase 1 — Core generation

- [x] FastAPI backend
- [x] User registration and login
- [x] JWT authentication
- [x] TripoSR inference integration
- [x] Hunyuan3D-2mini engine integration
- [x] GLB and OBJ export
- [x] Mesh cleanup
- [x] Generation status tracking
- [x] PostgreSQL integration
- [x] Neon database support
- [x] Alembic migrations

### Phase 2 — Mesh improvements

- [ ] UV unwrapping
- [ ] PBR texture baking
- [ ] Improved mesh repair
- [ ] Better topology optimization
- [ ] Automatic thumbnail generation
- [ ] Multi-view image reconstruction

### Phase 3 — Performance

- [ ] Durable background job queue
- [ ] GPU worker management
- [ ] Better VRAM monitoring
- [ ] Automatic GPU memory optimization
- [ ] Model caching improvements
- [ ] Job retry and recovery
- [ ] Inference performance benchmarking

### Phase 4 — Storage and deployment

- [ ] Cloudflare R2 or S3 object storage
- [ ] Persistent downloadable model URLs
- [ ] Docker Compose development setup
- [ ] CI/CD pipeline
- [ ] Production monitoring
- [ ] Rate limiting and stronger API security

### Phase 5 — User experience

- [ ] React-based frontend
- [ ] Interactive Three.js 3D viewer
- [ ] User asset library
- [ ] Generation progress interface
- [ ] Download and export controls
- [ ] Model comparison and preview tools

---

## 🤝 Contributing

RenderForge is open source, and contributions are welcome!

You can help improve:

- AI model integrations.
- GPU memory optimization.
- 3D mesh processing.
- Backend API development.
- Database performance.
- Documentation.
- Testing.
- Security and reliability.

### Contribution workflow

1. Fork the repository.
2. Clone your fork.
3. Create a feature branch.
4. Implement your changes.
5. Run the relevant tests.
6. Commit your changes.
7. Open a Pull Request.

Example:

```bash
git clone https://github.com/YOUR-USERNAME/renderforgebackend.git

cd renderforgebackend

git checkout -b feat/new-feature

git add .

git commit -m "feat: add new feature"

git push origin feat/new-feature
```

Then create a Pull Request against the original repository.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes.

For bugs and feature requests, use GitHub Issues.

---

## 🔒 Security

If you discover a security vulnerability:

- Do not publish API keys or passwords.
- Do not share private user information.
- Do not expose database credentials in public issues.
- Report vulnerabilities privately to the repository maintainer.

Never commit `.env` files or secret keys.

---

## 📜 License

RenderForge is intended to be distributed under the MIT License.

See the [LICENSE](LICENSE) file for the exact terms.

**Third-party model weights and dependencies have their own licenses.** Check the individual model repositories and their terms before redistributing weights or generated assets.

---

## 🙌 Acknowledgements

RenderForge builds on the work of the open-source community.

Special thanks to:

- [VAST-AI-Research / TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [Tencent / Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Trimesh](https://trimesh.org/)
- [Hugging Face](https://huggingface.co/)
- [Blender](https://www.blender.org/)

---

## ⭐ Support the Project

If you find RenderForge useful:

- Star the repository.
- Report bugs.
- Suggest new features.
- Contribute code.
- Share the project with other developers.

**Let's make local and open-source AI-powered 3D generation more accessible to everyone!**

Repository: https://github.com/vigneshuvn-21ITL133/renderforgebackend

