"""Central configuration and reproducible experiment design."""
from dataclasses import dataclass, asdict

EXPECTED_NODES = 4050
YEARS = (2019, 2022)
REPRESENTATIONS = ("UU", "UD", "WD")
FAMILIES = ("ER", "BA", "WS")
ETA_LEVELS = (0.5, 0.9, 1.0, 1.1, 1.5, 2.0)


@dataclass(frozen=True)
class ThresholdConfig:
    candidates: tuple = (0.0, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    local_sensitivity_factor: float = 3.0
    path_sample_sources: int = 256
    betweenness_k: int = 64


@dataclass(frozen=True)
class SyntheticConfig:
    realizations: int = 20
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
        return {"expected_nodes": self.expected_nodes,
                "threshold": asdict(self.threshold),
                "synthetic": asdict(self.synthetic)}
