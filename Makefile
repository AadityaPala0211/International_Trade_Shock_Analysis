PYTHON ?= python
OUT ?= results_network_analysis
DATA ?= data
THRESHOLD ?=
RHO ?= 0.2
BETA_UU ?= 0.1
BETA_UD ?= 0.1
BETA_WD ?= 0.1
MAX_STEPS ?= 300
TOLERANCE ?= 1e-8
SEED ?= 20261030

install:
	$(PYTHON) -m pip install -r code/network_structural_analysis/requirements.txt

test:
	PYTHONPATH=code $(PYTHON) -m unittest discover -s code/network_structural_analysis/tests -v

inspect:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage inspect --data-dir $(DATA)

threshold:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage threshold --data-dir $(DATA) --out $(OUT)

real:
	@test -n "$(THRESHOLD)" || (echo "Provide a reviewed threshold: make real THRESHOLD=0.0001" && exit 1)
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage real --data-dir $(DATA) --out $(OUT) --threshold $(THRESHOLD)

synthetic:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage synthetic --out $(OUT) --realizations 20 --seed $(SEED)

# Preflight stages inspect persisted matrices and calculate lambda_max and beta_c.
# They do not simulate SIS or modify the structural outputs.
dynamics-validate-real:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics-validate --scope real --out $(OUT) --rho $(RHO)

dynamics-validate-synthetic:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics-validate --scope synthetic --out $(OUT) --rho $(RHO)

dynamics-validate-all:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics-validate --scope all --out $(OUT) --rho $(RHO)

# Absolute beta values are illustrative defaults; choose them after reviewing preflight beta_c output.
dynamics-real:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics --scope real --out $(OUT) --seed $(SEED) --rho $(RHO) --beta-uu $(BETA_UU) --beta-ud $(BETA_UD) --beta-wd $(BETA_WD) --max-steps $(MAX_STEPS) --tolerance $(TOLERANCE)

dynamics-synthetic:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics --scope synthetic --out $(OUT) --seed $(SEED) --rho $(RHO) --beta-uu $(BETA_UU) --beta-ud $(BETA_UD) --beta-wd $(BETA_WD) --max-steps $(MAX_STEPS) --tolerance $(TOLERANCE)

dynamics-all:
	PYTHONPATH=code $(PYTHON) -m network_structural_analysis.run_pipeline --stage dynamics --scope all --out $(OUT) --seed $(SEED) --rho $(RHO) --beta-uu $(BETA_UU) --beta-ud $(BETA_UD) --beta-wd $(BETA_WD) --max-steps $(MAX_STEPS) --tolerance $(TOLERANCE)
