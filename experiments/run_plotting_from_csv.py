from utils.experiment_evaluations import plot_fBm_results_from_csv


def run_plotting_from_csv() -> None:
    from configs.VESDE.fBm_T32_H07 import get_config
    config = get_config()
    plot_fBm_results_from_csv(config)


if __name__ == "__main__":
    run_plotting_from_csv()
