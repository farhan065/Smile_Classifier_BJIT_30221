# Smile Classifier

Classifies a face photo as **Smiling** or **Not Smiling**. FastAPI web app with
a scikit-learn model, PostgreSQL persistence, and Docker deployment.

**Project ID:** 30221

---

## Quick start

Requires Docker Desktop.

```bash
cp .env.example .env
docker network create YSDTP_B5_AI_30221
docker compose up -d --build
```

Open <http://localhost:8000>

To stop: `docker compose down`

---

## Features

| Page | Description |
|------|-------------|
| Home | Model overview, framework rationale, live accuracy |
| Train | Multi-file upload with per-class labelling, retrains the model |
| Classify | Single-image prediction with confidence score |
| History | All past classifications with image, class, and timestamp |

---

## Stack

FastAPI · Jinja2 · scikit-learn · scikit-image · Pillow · SQLAlchemy ·
PostgreSQL 16 · Docker

---

## Model

A **Support Vector Machine (RBF kernel)** trained on **HOG** features.

HOG describes edge orientation rather than raw pixel intensity, capturing the
shape cues that distinguish a smile while remaining robust to lighting. An SVM
separates the two classes well in this high-dimensional feature space, and
scikit-learn keeps training fast and CPU-only — a deep network would add
significant complexity for no practical gain on ~1,200 images.

Preprocessing is shared between training and inference (`app/ml/preprocessing.py`)
so features can never drift between the two paths: grayscale → resize 64×64 →
normalise → HOG.

**Evaluation.** The full dataset uses a 20% held-out test set. Batches uploaded
through the web UI are too small for a reliable single split, so accuracy is
estimated with k-fold cross-validation, then the model is refitted on all data.
Metrics are written to `model/model_meta.json` and surfaced in the UI.

Current accuracy on the full dataset: **~93%**

---

## Configuration

Copy `.env.example` to `.env`.

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database user | `smile_user` |
| `POSTGRES_PASSWORD` | Database password | — |
| `POSTGRES_DB` | Database name | `smile_db` |
| `POSTGRES_HOST` | `localhost` locally; Compose overrides to `db` | `localhost` |
| `POSTGRES_PORT` | Database port | `5432` |
| `MAX_FILE_SIZE_MB` | Per-file upload limit | `5` |
| `MAX_TRAIN_FILES` | Files per training batch | `10` |
| `DISPLAY_TIMEZONE` | IANA timezone for display | `Asia/Dhaka` |

`.env` is git-ignored. Leave `POSTGRES_HOST=localhost` — `docker-compose.yml`
overrides it inside the container, so one file serves both environments.

---

## Local development

The database runs in Docker; the app runs on the host for hot reload.

```bash
docker compose up -d db
python -m venv venv && .\venv\Scripts\Activate.ps1   # or: source venv/bin/activate
pip install -r requirements.txt
python -m app.init_db
uvicorn app.main:app --reload
```

---

## Training

Place the [Kaggle dataset](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data)
in `data/` as `smile/`, `non_smile/`, `test/`, then:

```bash
python -m app.ml.train
```

Writes `model/smile_model.pkl` and `model/model_meta.json`.

The Train page retrains from uploaded images and **overwrites** the pickle.
Re-run the command above to restore the full-dataset model — the `./model`
bind mount makes it available to the running container immediately.

---

## Validation

Enforced server-side in `app/utils.py`:

- File size limited to `MAX_FILE_SIZE_MB`
- Training batches limited to `MAX_TRAIN_FILES`
- Only JPG, JPEG, PNG accepted — verified by decoding with Pillow, not by
  extension alone
- All accepted images converted to JPG before storage
- Classification blocked with a clear message when no trained model exists
- Training blocked unless both classes are present

---

## Architecture

```
network: YSDTP_B5_AI_30221
   ├── smile_db_30221    postgres:16, healthcheck: pg_isready
   └── smile_app_30221   FastAPI, depends_on db (healthy), 8000:8000
```

The app resolves PostgreSQL by service name `db` over the user-defined network.
`depends_on: condition: service_healthy` gates startup on the database actually
accepting connections, not merely having started.

**Data.** Classification records persist to a named volume (`smile_db_data`).
The trained model and uploaded images are bind-mounted to the host.

**Image.** `python:3.12-slim` base; dependencies installed before source is
copied so code changes don't invalidate the dependency layer; runs as a
non-root user; `data/` and `.env` excluded via `.dockerignore`.

---

## Linux ARM

The source is architecture-neutral — the target is chosen at build time.

On ARM hardware (Apple Silicon, ARM servers), the standard command already
produces a native ARM image:

```bash
docker compose up -d --build
```

To cross-build from x86:

```bash
docker buildx build --platform linux/arm64 -t smile-classifier-30221:1.0-arm64 --load .
docker inspect smile-classifier-30221:1.0-arm64 --format "{{.Architecture}}"
```

Emulated builds are slower than native — expected, not an error.

`platform:` is intentionally not pinned in `docker-compose.yml` so each host
builds for itself.

### Shipping a pre-built image

```bash
docker tag smile-classifier-30221:1.0-arm64 smile-classifier-30221:latest
docker save -o smile-classifier-30221-arm64.tar \
  smile-classifier-30221:1.0-arm64 smile-classifier-30221:latest
```

Recipient:

```bash
docker load -i smile-classifier-30221-arm64.tar
cp .env.example .env
docker network create YSDTP_B5_AI_30221
docker compose up -d          # no --build; uses the loaded image
```

---

## Project layout

```
app/
├── main.py            FastAPI routes and template filters
├── config.py          Configuration from environment
├── database.py        SQLAlchemy engine and session
├── models.py          ORM models
├── utils.py           Upload validation and JPG conversion
├── init_db.py         Table creation
├── ml/
│   ├── preprocessing.py   Shared feature extraction
│   ├── train.py           Training, metrics, persistence
│   └── inference.py       Model loading and prediction
├── templates/         Jinja2 templates
└── static/            CSS and runtime uploads

model/                 Trained artifacts (generated)
data/                  Dataset (local training only)
```

### Database schema

`history` — `id`, `image_path`, `predicted_class`, `created_at`

Timestamps are stored in UTC and rendered in `DISPLAY_TIMEZONE`.

---

## Commands

```bash
docker compose up -d --build              # build and start
docker compose down                       # stop
docker compose logs -f app                # app logs
docker ps                                 # container status

python -m app.ml.train                    # train on full dataset
python -m app.ml.inference <image>        # predict from CLI
python -m app.init_db                     # create tables

docker exec -it smile_db_30221 psql -U smile_user -d smile_db
```

---

## Troubleshooting

**`could not translate host name "db"`** — running locally with container
config. Set `POSTGRES_HOST=localhost` in `.env`.

**`Required environment variable 'X' is not set`** — `.env` missing or
incomplete. Copy from `.env.example`; no spaces around `=`.

**`network YSDTP_B5_AI_30221 not found`** — run
`docker network create YSDTP_B5_AI_30221`.

**Container restarting** — check `docker compose logs app`.

**Port in use** — change the host side of the mapping in `docker-compose.yml`.

**Stale content** — rebuild with `--build`, then hard-refresh (`Ctrl+Shift+R`).
