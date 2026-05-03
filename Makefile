IMAGE_NAME ?= wd14-tagger:cpu

.PHONY: test build-cpu

test:
	. .venv/bin/activate && python -m pytest -q

build-cpu:
	docker build -t $(IMAGE_NAME) -f Dockerfile.cpu .
