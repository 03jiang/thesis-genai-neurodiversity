

import math
from dataclasses import dataclass

from statsmodels.stats.power import GofChisquarePower


@dataclass(frozen=True)
class PowerAnalysisConfig:
    rows: int = 6
    columns: int = 3
    cramers_v: float = 0.26
    alpha: float = 0.05
    target_power: float = 0.80


def validate_config(config: PowerAnalysisConfig) -> None:
    if config.rows < 2 or config.columns < 2:
        raise ValueError("The contingency table must have at least 2 rows and 2 columns.")
    if not 0 < config.cramers_v <= 1:
        raise ValueError("Cramér's V must be between 0 and 1.")
    if not 0 < config.alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not 0 < config.target_power < 1:
        raise ValueError("Target power must be between 0 and 1.")


def calculate_required_sample_size(
    config: PowerAnalysisConfig,
) -> dict[str, float | int]:
    validate_config(config)

    min_dimension = min(config.rows - 1, config.columns - 1)
    degrees_of_freedom = (config.rows - 1) * (config.columns - 1)

    # Convert Cramér's V to Cohen's w.
    cohens_w = config.cramers_v * math.sqrt(min_dimension)

    # For a chi-square test with df degrees of freedom, statsmodels represents
    # this through n_bins = df + 1.
    required_n = GofChisquarePower().solve_power(
        effect_size=cohens_w,
        nobs=None,
        alpha=config.alpha,
        power=config.target_power,
        n_bins=degrees_of_freedom + 1,
    )

    return {
        "rows": config.rows,
        "columns": config.columns,
        "degrees_of_freedom": degrees_of_freedom,
        "min_dimension": min_dimension,
        "cramers_v": config.cramers_v,
        "cohens_w": cohens_w,
        "alpha": config.alpha,
        "target_power": config.target_power,
        "required_n_raw": required_n,
        "required_n_ceiling": math.ceil(required_n),
    }


def main() -> None:
    config = PowerAnalysisConfig()
    results = calculate_required_sample_size(config)

    print("=" * 60)
    print("Q5 CONTINGENCY-TABLE POWER ANALYSIS")
    print("=" * 60)
    print(
        f"Table structure: "
        f"{results['rows']} x {results['columns']}"
    )
    print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    print(f"Target Cramér's V: {results['cramers_v']:.3f}")
    print(
        "Converted Cohen's w: "
        f"{results['cohens_w']:.4f}"
    )
    print(f"Alpha: {results['alpha']:.2f}")
    print(f"Target power: {results['target_power']:.2f}")
    print(
        "Estimated required sample size: "
        f"{results['required_n_raw']:.2f}"
    )
    print(
        "Required sample size after rounding up: "
        f"{results['required_n_ceiling']}"
    )
    print("=" * 60)
    print(
        "Interpretation: under these assumptions, approximately "
        f"{results['required_n_ceiling']} respondents would be required."
    )


if __name__ == "__main__":
    main()