"""Central experiment configuration.

The active research pipeline is WD-only. Legacy multi-representation constants
are explicitly named LEGACY_* and are used only by compatibility helpers and
historical tests; they are not exposed by the active CLI or new manifests.
"""
from dataclasses import dataclass, asdict

EXPECTED_NODES = 4050
YEARS = (2019, 2022)
ACTIVE_REPRESENTATION = "WD"
REPRESENTATIONS = ("WD",)
LEGACY_REPRESENTATIONS = ("UU", "UD", "WD")
FAMILIES = ("ER", "BA", "WS")
ETA_LEVELS = (0.5, 0.9, 1.0, 1.1, 1.5, 2.0)
STAGE1_ETA_LEVELS = (0.9, 1.1)
ACTIVE_SYNTHETIC_REALIZATIONS = 1
ACTIVE_RHO = 0.2


@dataclass(frozen=True)
class ThresholdConfig:
    candidates: tuple = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    local_sensitivity_factor: float = 3.0
    path_sample_sources: int = 256
    betweenness_k: int = 64


@dataclass(frozen=True)
class SyntheticConfig:
    # Retained for legacy helper compatibility; the active workflow hard-codes
    # exactly one realization per synthetic parameter configuration.
    realizations: int = 10
    seed: int = 20261030
    ws_rewire_probabilities: tuple = (0.01, 0.10, 0.50)
    mean_degree_multipliers: tuple = (0.5, 1.0, 1.5)
    ba_multipliers: tuple = (0.5, 1.0, 1.5)
    weight_distribution_sample_cap: int = 2_000_000


@dataclass(frozen=True)
class PipelineConfig:
    expected_nodes: int = EXPECTED_NODES
    threshold: ThresholdConfig = ThresholdConfig()
    synthetic: SyntheticConfig = SyntheticConfig()

    def to_dict(self):
        return {
            "expected_nodes": self.expected_nodes,
            "threshold": asdict(self.threshold),
            "synthetic": asdict(self.synthetic),
            "active_representation": ACTIVE_REPRESENTATION,
            "active_synthetic_realizations": ACTIVE_SYNTHETIC_REALIZATIONS,
            "active_rho": ACTIVE_RHO,
            "stage1_eta_levels": list(STAGE1_ETA_LEVELS),
        }
