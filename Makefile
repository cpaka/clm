.PHONY: deploy redeploy upload-dataset logs train-parallel train-shards train-incremental ingest warmup bump-version fetch-corpus

# One-time GPU kernel warm-up: compiles cupy CUDA kernels and caches them on the Volume.
# Run this ONCE before the first `make train-parallel`.
warmup:
	modal run modal_app.py::warmup_cupy

# Upload the raw Wikipedia dataset to the Modal Volume (one-time, shared across all versions).
# Requires Kaggle credentials in the 'kaggle-credentials' Modal secret.
upload-dataset:
	modal run modal_app.py::upload_dataset

upload-dataset-force:
	modal run modal_app.py::upload_dataset --force

# ── Version management ─────────────────────────────────────────────────────────
# Each bump doubles the corpus (v4→v4.1: 1M→2M chars, 5K→10K seqs, etc.)
#
#   make bump-version        # edit modal_app.py: increment VERSION + corpus size
#   make fetch-corpus        # sample the new (larger) corpus from the dataset
#   make train-parallel      # retrain on larger corpus
#   make deploy              # push new version to Modal
#
# Or in one shot:  make bump-version fetch-corpus train-parallel deploy
bump-version:
	python3 bump_version.py

bump-version-dry:
	python3 bump_version.py --dry-run

# Sample a corpus slice from the shared raw dataset (uses current CORPUS_CONFIG).
fetch-corpus:
	modal run modal_app.py::fetch_corpus

# ── Deploy / train ─────────────────────────────────────────────────────────────

# Full deploy: push code + auto-train (idempotent per VERSION).
deploy:
	modal deploy modal_app.py
	modal run modal_app.py::bootstrap

# Force retrain even if a checkpoint exists for this VERSION.
redeploy:
	modal deploy modal_app.py
	modal run modal_app.py::bootstrap --force

# Train voting units in parallel containers, assemble ensemble.
train-parallel:
	modal run modal_app.py::train_parallel

# Incremental training: 3 corpus batches, model queryable after each batch.
# The model improves in-place — no redeploy needed between batches.
train-incremental:
	modal run modal_app.py::train_incremental

# Train N corpus shards in parallel, merge segments.
train-shards:
	modal run modal_app.py::train_shards

# Run one incremental ingest pass (normally triggered by cron at 03:00 UTC).
ingest:
	modal run modal_app.py::ingest

# Tail logs for the running web app.
logs:
	modal app logs clm-chat-$(shell grep '^VERSION' modal_app.py | cut -d'"' -f2)
