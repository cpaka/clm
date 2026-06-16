.PHONY: deploy redeploy logs

# Full deploy: push code to Modal, then run corpus fetch + training in one go.
# Idempotent: bootstrap skips if this version's model.pkl already exists.
deploy:
	modal deploy modal_app.py
	modal run modal_app.py::bootstrap

# Force retrain even if model already exists for this version.
redeploy:
	modal deploy modal_app.py
	modal run modal_app.py::bootstrap --force

# Tail logs for the running web app.
logs:
	modal app logs cglm-chat-$(shell grep '^VERSION' modal_app.py | cut -d'"' -f2)
