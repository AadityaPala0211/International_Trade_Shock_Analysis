"""Run all quick, saved-output-only presentation plots."""
from plot_structural_points import main as structural
from plot_fixed_eta_trajectory import main as trajectory
from plot_sis_response_vs_eta import main as sis_response
from plot_rank_stability import main as rank_stability
from plot_synthetic_benchmarks import main as synthetic_benchmarks
from plot_top_hubs import main as top_hubs


def main():
    for plot in (structural, trajectory, sis_response, rank_stability,
                 synthetic_benchmarks, top_hubs):
        plot()


if __name__ == "__main__":
    main()
