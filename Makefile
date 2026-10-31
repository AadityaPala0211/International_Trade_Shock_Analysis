PYTHON ?= python
DATA ?= data
OUT ?= results/revised_structural_analysis
CENSUS_OUT ?= results/single_node_shock_census
TARGETED_OUT ?= results/targeted_individual_node_sis
THRESHOLD ?= 0.001
RHO ?= 0.2
SEED ?= 20261030
BATCH_SIZE ?= 128
MAX_STEPS ?= 300

install:
	$(PYTHON) -m pip install -r code/network_structural_analysis/requirements.txt

test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s code/network_structural_analysis/tests -v

compile:
	PYTHONPATH=code $(PYTHON) -m compileall -q code/network_structural_analysis

preflight:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage preflight --data-dir $(DATA) --expected-nodes 4050

structural-real:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage structural --scope real --data-dir $(DATA) --out $(OUT) --threshold $(THRESHOLD) --expected-nodes 4050 --seed $(SEED)

validate-real:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage validate-structural --scope real --out $(OUT) --expected-nodes 4050

structural-synthetic:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage structural --scope synthetic --out $(OUT) --expected-nodes 4050 --seed $(SEED)

validate-synthetic:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage validate-structural --scope synthetic --out $(OUT) --expected-nodes 4050

validate-all:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage validate-structural --scope all --out $(OUT) --expected-nodes 4050

dynamics-preflight:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics-preflight --out $(OUT) --census-out $(CENSUS_OUT) --rho $(RHO)

census:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage census --out $(OUT) --census-out $(CENSUS_OUT) --rho $(RHO) --batch-size $(BATCH_SIZE) --max-steps $(MAX_STEPS)

census-validate:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage census-validate --out $(OUT) --census-out $(CENSUS_OUT)

compare-select:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage compare-select --out $(OUT) --census-out $(CENSUS_OUT)

targeted:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage targeted --out $(OUT) --census-out $(CENSUS_OUT) --targeted-out $(TARGETED_OUT) --rho $(RHO) --batch-size $(BATCH_SIZE) --max-steps $(MAX_STEPS)

targeted-validate:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage targeted-validate --targeted-out $(TARGETED_OUT)
