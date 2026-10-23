.PHONY: install test lint run
install:
	python -m pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check src modal_app.py modal_models.py scripts tests
run:
	uvicorn multimodal_embeddings_ai_inference.api:local_app --factory --host 127.0.0.1 --port 8000
