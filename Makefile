.PHONY: deploy redeploy upload-dataset logs train-parallel train-shards ingest

# Upload the raw Wikipedia dataset to the Modal Volume (one-time, shared across all versions).
# Safe to re-run: skips download if the file already exists.
# Requires Kaggle credentials in the 'kaggle-credentials' Modal secret.
# Before first run: accept the dataset license at
#   https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish
upload-dataset:
	modal run modal_app.py::upload_dataset

# Force re-download even if dataset file already exists.
upload-dataset-force:
	modal run modal_app.py::upload_dataset --force

# Full deploy: push code to Modal, then auto-train (idempotent per VERSION).
# If the raw dataset is already in the volume, bootstrap skips the download step.
deploy:
	modal deploy modal_app.py
	modal run modal_app.py::bootstrap

# Force retrain even if model.pkl already exists for this VERSION.
redeploy:
	modal deploy modal_app.py
	modal run modal_app.py::bootstrap --force

# Train the voting units in parallel containers, then assemble + save the
# ensemble (NEXT_STEPS #1). Requires a corpus already on the Volume (deploy first).
train-parallel:
	modal run modal_app.py::train_parallel

# Train N shards in parallel and merge segments for maximum corpus coverage (#2).
# Increase n_shards to scale data throughput linearly.
train-shards:
	modal run modal_app.py::train_shards

# Run one incremental ingest pass (normally triggered by cron at 03:00 UTC, #6).
ingest:
	modal run modal_app.py::ingest

# Tail logs for the running web app.
logs:
	modal app logs clm-chat-$(shell grep '^VERSION' modal_app.py | cut -d'"' -f2)
