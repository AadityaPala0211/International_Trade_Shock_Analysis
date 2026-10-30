PYTHON ?= python

install:
	$(PYTHON) -m pip install -r code/network_structural_analysis/requirements.txt

test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s code/network_structural_analysis/tests -v

inspect:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --data-dir data --out results_network_analysis/inspection --inspect-only

threshold:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --data-dir data --out results_network_analysis/threshold_review --threshold-only

run:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --data-dir data --out results_network_analysis --realizations 20
