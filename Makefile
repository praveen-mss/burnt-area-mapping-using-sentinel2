PYTHON ?= python

.PHONY: train daily-inference reprocess aggregate test

train:
	$(PYTHON) scripts/run_training.py

daily-inference:
	$(PYTHON) scripts/run_daily_inference.py

reprocess:
	$(PYTHON) scripts/run_reprocess_all.py

aggregate:
	$(PYTHON) scripts/run_aggregation.py

test:
	$(PYTHON) -m unittest discover -s tests
