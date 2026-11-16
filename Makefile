.PHONY: help install install-cpu lint test smoke tokenizer tokenize train

help:
	@echo "install      : create .venv and install dev dependencies (CPU torch)"
	@echo "lint         : run ruff"
	@echo "test         : run pytest"
	@echo "smoke        : run a 50-step synthetic training run"
	@echo "tokenizer    : train SentencePiece tokenizer (see docs/01-data.md)"
	@echo "tokenize     : encode corpus into uint16 shards (see docs/01-data.md)"
	@echo "train        : run pretraining (CONFIG=configs/model-60m.yaml)"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest -q

smoke:
	.venv/bin/python -m synax.train.pretrain --config configs/tiny.yaml --synthetic --steps 50

tokenizer:
	.venv/bin/python -m synax.data.tokenizer train \
		--input $(INPUT) --output-prefix $(OUTPUT) --vocab-size $(VOCAB)

tokenize:
	.venv/bin/python -m synax.data.tokenizer encode \
		--input $(INPUT) --tokenizer $(TOKENIZER) \
		--output $(TRAIN_BIN) --val-output $(VAL_BIN)

train:
	.venv/bin/python -m synax.train.pretrain --config $(CONFIG)
