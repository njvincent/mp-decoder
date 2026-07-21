"""Shared loading, plotting, and fitting helpers for analysis notebooks.

The simulation result files are small Julia-style ``name = value`` text files.
Keeping their parsing and aggregation here makes each notebook describe only the
scan it is analysing instead of reimplementing the same bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_distribution


DEFAULT_CONTROL_TARGET_YLIM = (1e-3, 1)
DEFAULT_FIT_P_RANGE = (0.011, 0.020)
DEFAULT_FIT_MIN_L = 5


@dataclass(frozen=True)
class RateEstimate:
    """A rate grid and its binomial standard error."""

    values: np.ndarray
    errors: np.ndarray


@dataclass(frozen=True)
class FailureScan:
    """Aggregated failure-rate data indexed by lattice size and error rate."""

    data_dir: Path
    p_values: np.ndarray
    L_values: np.ndarray
    success_probability: np.ndarray
    trials: np.ndarray
    repeat_counts: np.ndarray
    rates: Mapping[str, RateEstimate]
    matched_file_count: int
    ignored_file_count: int
    scan_label: str

    def rate(self, name: str = "overall") -> RateEstimate:
        """Return one named logical-error rate (``overall`` by default)."""

        try:
            return self.rates[name]
        except KeyError as error:
            available = ", ".join(sorted(self.rates))
            raise KeyError(f"Unknown rate {name!r}; available rates: {available}") from error

    def fit_arrays(self, name: str = "overall") -> tuple[np.ndarray, ...]:
        """Return flattened ``p, L, rate, error`` arrays with usable errors."""

        estimate = self.rate(name)
        p_grid, L_grid = np.meshgrid(self.p_values, self.L_values)
        mask = (
            np.isfinite(estimate.values)
            & np.isfinite(estimate.errors)
            & (estimate.errors > 0)
        )
        return (
            p_grid[mask],
            L_grid[mask],
            estimate.values[mask],
            estimate.errors[mask],
        )

    def print_summary(self) -> None:
        """Print the compact load summary used throughout the notebooks."""

        print(
            f"Loaded {self.matched_file_count} requested {self.scan_label} files; "
            f"ignored {self.ignored_file_count} files outside the requested lists"
        )
        print("L values:", self.L_values)
        print("p values:", self.p_values)
        print("repeat counts:\n", self.repeat_counts)


@dataclass(frozen=True)
class TrelScan:
    """Aggregated relaxation-time statistics on an ``(L, p)`` grid."""

    data_dir: Path
    p_values: np.ndarray
    L_values: np.ndarray
    mean: np.ndarray
    error: np.ndarray
    std: np.ndarray
    maximum: np.ndarray
    samples: np.ndarray
    matched_file_count: int
    ignored_file_count: int

    def print_summary(self) -> None:
        print(
            f"Loaded {self.matched_file_count} requested trel files; "
            f"ignored {self.ignored_file_count} files outside the requested lists"
        )
        print("trel L values:", self.L_values)
        print("trel p values:", self.p_values)
        print("trel sample counts:\n", self.samples.astype(int))


@dataclass(frozen=True)
class FiniteSizeScalingFit:
    """Result of a polynomial finite-size scaling fit."""

    label: str
    polynomial_degree: int
    parameters: np.ndarray
    parameter_errors: np.ndarray
    fit_p_range: tuple[float, float]
    min_L: int
    p_fit: np.ndarray
    L_fit: np.ndarray
    rate_fit: np.ndarray
    rate_error_fit: np.ndarray
    chi2: float
    dof: int
    p_value: float

    @property
    def pc(self) -> float:
        return float(self.parameters[0])

    @property
    def pc_error(self) -> float:
        return float(self.parameter_errors[0])

    @property
    def nu(self) -> float:
        return float(self.parameters[1])

    @property
    def nu_error(self) -> float:
        return float(self.parameter_errors[1])

    @property
    def chi2_per_dof(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else np.nan


@dataclass(frozen=True)
class ScalingAnsatzFit:
    """Result from one of the exploratory polynomial scaling ansatzes."""

    polynomial_order: int
    include_finite_size_correction: bool
    parameters: np.ndarray
    covariance: np.ndarray
    weighted_absolute_residual: float
    solver_message: str
    solver_status: int

    @property
    def parameter_errors(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance))


def _parse_vector(text: str, name: str, path: Path) -> np.ndarray:
    match = re.search(rf"^{re.escape(name)}\s*=\s*\[([^\]]+)\]\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not parse {name} from {path}")
    values = np.fromstring(match.group(1), sep=" ")
    if values.size == 0:
        raise ValueError(f"Parsed empty {name} from {path}")
    return values


def _parse_scalar(text: str, name: str, path: Path) -> float:
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*(?:\[([^\]]+)\]|([^\s#]+))\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not parse {name} from {path}")
    value_text = match.group(1) or match.group(2)
    values = np.fromstring(value_text, sep=" ")
    return float(values[0]) if values.size else float(value_text)


def _validated_grid_values(
    p_values: Sequence[float], L_values: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    p_array = np.asarray(p_values, dtype=float)
    L_array = np.asarray(L_values, dtype=int)
    if p_array.ndim != 1 or L_array.ndim != 1 or not p_array.size or not L_array.size:
        raise ValueError("p_values and L_values must be non-empty one-dimensional sequences")
    if np.unique(p_array).size != p_array.size or np.unique(L_array).size != L_array.size:
        raise ValueError("p_values and L_values must not contain duplicates")
    return p_array, L_array


def _grid_index(
    cur_l: int, cur_p: float, L_values: np.ndarray, p_values: np.ndarray
) -> tuple[int, int] | None:
    l_indices = np.flatnonzero(L_values == cur_l)
    p_indices = np.flatnonzero(np.isclose(p_values, cur_p))
    if not l_indices.size or not p_indices.size:
        return None
    return int(l_indices[0]), int(p_indices[0])


def _binomial_estimate(failures: np.ndarray, trials: np.ndarray) -> RateEstimate:
    values = np.divide(
        failures,
        trials,
        out=np.full_like(failures, np.nan, dtype=float),
        where=trials > 0,
    )
    variance = np.divide(
        values * (1 - values),
        trials,
        out=np.full_like(values, np.nan),
        where=trials > 0,
    )
    return RateEstimate(values=values, errors=np.sqrt(variance))


def load_ft_scan(
    data_dir: str | Path,
    *,
    p_values: Sequence[float],
    L_values: Sequence[int],
    expected_repeats: int | None,
    result_pattern: str = "2d_CNOT_*_Ft*.txt",
    success_field: str = "CNOT_Ft",
    failure_fields: Mapping[str, str] | None = None,
    scan_label: str = "CNOT Ft",
) -> FailureScan:
    """Load and repeat-weight an Ft scan.

    ``failure_fields`` maps notebook-facing rate names (for example
    ``"control"``) to count fields in the result files.
    """

    data_dir = Path(data_dir)
    p_values, L_values = _validated_grid_values(p_values, L_values)
    failure_fields = dict(failure_fields or {})
    result_paths = sorted(data_dir.rglob(result_pattern))
    if not result_paths:
        raise FileNotFoundError(
            f"No {scan_label} result files found under {data_dir.resolve()}. "
            "Run the corresponding scan first, or change data_dir."
        )

    shape = (len(L_values), len(p_values))
    successes = np.zeros(shape)
    trials = np.zeros(shape)
    repeat_counts = np.zeros(shape, dtype=int)
    component_failures = {name: np.zeros(shape) for name in failure_fields}
    matched_file_count = 0

    for path in result_paths:
        text = path.read_text()
        cur_l_float = _parse_scalar(text, "L", path)
        cur_l = int(cur_l_float)
        if cur_l != cur_l_float:
            raise ValueError(f"Expected integer L in {path}, got {cur_l_float}")
        cur_p = _parse_scalar(text, "p", path)
        index = _grid_index(cur_l, cur_p, L_values, p_values)
        if index is None:
            continue

        ft = _parse_scalar(text, success_field, path)
        cur_trials = _parse_scalar(text, "trials", path)
        if cur_trials < 0:
            raise ValueError(f"Expected non-negative trials in {path}, got {cur_trials}")

        i, j = index
        successes[i, j] += ft * cur_trials
        trials[i, j] += cur_trials
        repeat_counts[i, j] += 1
        for rate_name, field_name in failure_fields.items():
            component_failures[rate_name][i, j] += _parse_scalar(text, field_name, path)
        matched_file_count += 1

    if matched_file_count == 0:
        raise ValueError(
            f"No {scan_label} result files matched the requested L_values/p_values."
        )

    missing_mask = repeat_counts == 0
    if np.any(missing_mask):
        missing = [
            (int(L_values[i]), float(p_values[j]))
            for i, j in np.argwhere(missing_mask)
        ]
        print(
            f"Warning: missing requested {scan_label} data for these (L, p) points; "
            f"leaving them as NaN: {missing}"
        )

    nonzero_repeat_counts = repeat_counts[repeat_counts > 0]
    if np.unique(nonzero_repeat_counts).size > 1:
        print("Warning: not every (L, p) point has the same number of repeats.")
    if expected_repeats is not None and np.any(nonzero_repeat_counts != expected_repeats):
        print(f"Warning: expected {expected_repeats} repeats per (L, p) point.")

    success_probability = np.divide(
        successes,
        trials,
        out=np.full_like(successes, np.nan),
        where=trials > 0,
    )
    rates: dict[str, RateEstimate] = {
        "overall": _binomial_estimate(trials - successes, trials)
    }
    rates.update(
        {
            name: _binomial_estimate(failures, trials)
            for name, failures in component_failures.items()
        }
    )

    return FailureScan(
        data_dir=data_dir,
        p_values=p_values,
        L_values=L_values,
        success_probability=success_probability,
        trials=trials,
        repeat_counts=repeat_counts,
        rates=rates,
        matched_file_count=matched_file_count,
        ignored_file_count=len(result_paths) - matched_file_count,
        scan_label=scan_label,
    )


def load_trel_scan(
    data_dir: str | Path,
    *,
    p_values: Sequence[float],
    L_values: Sequence[int],
    result_pattern: str = "2d_trel_*.txt",
) -> TrelScan:
    """Load aggregate relaxation-time result files."""

    data_dir = Path(data_dir)
    p_values, L_values = _validated_grid_values(p_values, L_values)
    result_paths = sorted(data_dir.glob(result_pattern))
    if not result_paths:
        raise FileNotFoundError(f"No trel result files found under {data_dir.resolve()}.")

    shape = (len(L_values), len(p_values))
    mean = np.full(shape, np.nan)
    error = np.full(shape, np.nan)
    std = np.full(shape, np.nan)
    maximum = np.full(shape, np.nan)
    samples = np.zeros(shape)
    record_counts = np.zeros(shape, dtype=int)
    ignored_file_count = 0

    for path in result_paths:
        text = path.read_text()
        cur_l = int(_parse_scalar(text, "L", path))
        cur_p = _parse_scalar(text, "p", path)
        index = _grid_index(cur_l, cur_p, L_values, p_values)
        if index is None:
            ignored_file_count += 1
            continue

        trel_stats = _parse_vector(text, "trel_stats", path)
        cur_samples = _parse_vector(text, "samps", path)[0]
        if trel_stats.size < 2:
            raise ValueError(f"Expected trel_stats to contain at least mean/std in {path}")
        if cur_samples <= 0:
            raise ValueError(f"Expected positive samps in {path}, got {cur_samples}")

        i, j = index
        if record_counts[i, j] != 0:
            raise ValueError(f"Duplicate aggregate trel result for L={cur_l}, p={cur_p}")
        mean[i, j] = trel_stats[0]
        std[i, j] = trel_stats[1]
        maximum[i, j] = trel_stats[2] if trel_stats.size > 2 else np.nan
        samples[i, j] = cur_samples
        error[i, j] = trel_stats[1] / np.sqrt(cur_samples)
        record_counts[i, j] = 1

    missing_mask = record_counts == 0
    if np.any(missing_mask):
        missing = [
            (int(L_values[i]), float(p_values[j]))
            for i, j in np.argwhere(missing_mask)
        ]
        raise ValueError(f"Missing trel data for these (L, p) points: {missing}")

    return TrelScan(
        data_dir=data_dir,
        p_values=p_values,
        L_values=L_values,
        mean=mean,
        error=error,
        std=std,
        maximum=maximum,
        samples=samples,
        matched_file_count=len(result_paths) - ignored_file_count,
        ignored_file_count=ignored_file_count,
    )


def plot_rate_scan(
    scan: FailureScan,
    *,
    title: str,
    rate_name: str = "overall",
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot one logical-error rate for every lattice size in a scan."""

    estimate = scan.rate(rate_name)
    _, ax = plt.subplots(figsize=(5, 5))
    for l_idx, cur_l in enumerate(scan.L_values):
        ax.errorbar(
            scan.p_values,
            estimate.values[l_idx],
            yerr=estimate.errors[l_idx],
            fmt="o-",
            capsize=3,
            label=f"L = {cur_l}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel(ylabel or f"{rate_name}_error_rate")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_trel_scan(scan: TrelScan, *, title: str = "Baseline trel") -> None:
    """Plot relaxation time for every lattice size in a scan."""

    _, ax = plt.subplots(figsize=(5, 5))
    for l_idx, cur_l in enumerate(scan.L_values):
        ax.errorbar(
            scan.p_values,
            scan.mean[l_idx],
            yerr=scan.error[l_idx],
            fmt="o-",
            capsize=3,
            label=f"L = {cur_l}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("t_rel")
    ax.set_title(title)
    ax.legend()
    plt.show()


def finite_size_scaling_model(
    pL: tuple[np.ndarray, np.ndarray],
    pc: float,
    nu: float,
    A: float,
    *coefficients: float,
) -> np.ndarray:
    """Polynomial expansion in ``x = (p - pc) L**(1/nu)``.

    ``A`` is the constant term and ``coefficients[k - 1]`` multiplies
    ``x**k``. The fit wrapper supplies at least one non-constant coefficient.
    """

    p, L = pL
    x = (p - pc) * L ** (1 / nu)
    return A + sum(
        coefficient * x**power
        for power, coefficient in enumerate(coefficients, 1)
    )


def _validate_polynomial_degree(polynomial_degree: int) -> int:
    if (
        isinstance(polynomial_degree, bool)
        or not isinstance(polynomial_degree, (int, np.integer))
        or polynomial_degree < 1
    ):
        raise ValueError("polynomial_degree must be a positive integer")
    return int(polynomial_degree)


def _polynomial_coefficient_label(power: int) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if power < len(labels):
        return labels[power]
    return f"c_{{{power}}}"


def fit_finite_size_scaling(
    scan: FailureScan,
    *,
    label: str,
    rate_name: str = "overall",
    fit_p_range: tuple[float, float] = DEFAULT_FIT_P_RANGE,
    min_L: int = DEFAULT_FIT_MIN_L,
    polynomial_degree: int = 2,
) -> FiniteSizeScalingFit:
    """Fit a polynomial scaling model of configurable positive degree."""

    polynomial_degree = _validate_polynomial_degree(polynomial_degree)
    estimate = scan.rate(rate_name)
    p_grid, L_grid = np.meshgrid(scan.p_values, scan.L_values)
    fit_mask = (
        np.isfinite(estimate.values)
        & np.isfinite(estimate.errors)
        & (estimate.errors > 0)
        & (p_grid >= fit_p_range[0])
        & (p_grid <= fit_p_range[1])
        & (L_grid >= min_L)
    )
    p_fit = p_grid[fit_mask]
    L_fit = L_grid[fit_mask]
    rate_fit = estimate.values[fit_mask]
    rate_error_fit = estimate.errors[fit_mask]
    parameter_count = 3 + polynomial_degree
    if rate_fit.size < parameter_count:
        raise ValueError(
            f"A degree-{polynomial_degree} fit has {parameter_count} parameters and "
            f"needs at least that many usable points; found {rate_fit.size}"
        )

    initial_coefficients = ([5.0, 100.0] + [0.0] * polynomial_degree)[
        :polynomial_degree
    ]
    initial_parameters = [
        np.mean(fit_p_range),
        1.3,
        np.median(rate_fit),
        *initial_coefficients,
    ]
    lower_bounds = [fit_p_range[0], 0.2] + [-np.inf] * (1 + polynomial_degree)
    upper_bounds = [fit_p_range[1], 10.0] + [np.inf] * (1 + polynomial_degree)
    parameters, covariance = curve_fit(
        finite_size_scaling_model,
        (p_fit, L_fit),
        rate_fit,
        p0=initial_parameters,
        sigma=rate_error_fit,
        absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds),
        maxfev=500000,
    )
    parameter_errors = np.sqrt(np.diag(covariance))
    residual = (
        rate_fit - finite_size_scaling_model((p_fit, L_fit), *parameters)
    ) / rate_error_fit
    chi2 = float(np.sum(residual**2))
    dof = int(rate_fit.size - parameters.size)
    p_value = float(chi2_distribution.sf(chi2, dof)) if dof > 0 else np.nan

    return FiniteSizeScalingFit(
        label=label,
        polynomial_degree=polynomial_degree,
        parameters=parameters,
        parameter_errors=parameter_errors,
        fit_p_range=fit_p_range,
        min_L=min_L,
        p_fit=p_fit,
        L_fit=L_fit,
        rate_fit=rate_fit,
        rate_error_fit=rate_error_fit,
        chi2=chi2,
        dof=dof,
        p_value=p_value,
    )


def format_parenthetical_uncertainty(value: float, uncertainty: float) -> str:
    """Format ``value(error)`` with one significant uncertainty digit."""

    if not np.isfinite(value) or not np.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:g}"
    uncertainty_exponent = int(np.floor(np.log10(abs(uncertainty))))
    decimal_places = max(0, -uncertainty_exponent)
    uncertainty_digits = int(np.floor(uncertainty * 10**decimal_places + 0.5))
    if uncertainty_digits == 10:
        decimal_places = max(0, decimal_places - 1)
        uncertainty_digits = int(np.floor(uncertainty * 10**decimal_places + 0.5))
    return f"{value:.{decimal_places}f}({uncertainty_digits})"


def display_finite_size_scaling_fit(fit: FiniteSizeScalingFit, ylabel: str) -> None:
    """Display a fit table, quality warning, and scaling-collapse plot."""

    pc, nu, A, *coefficients = fit.parameters
    pc_error, nu_error, A_error, *coefficient_errors = fit.parameter_errors
    p_min, p_max = fit.fit_p_range
    included_L = sorted(np.unique(fit.L_fit).astype(int))
    polynomial_terms = ["A"]
    coefficient_rows = [
        f"| $A$ | {format_parenthetical_uncertainty(A, A_error)} |"
    ]
    for power, (coefficient, coefficient_error) in enumerate(
        zip(coefficients, coefficient_errors), 1
    ):
        coefficient_label = _polynomial_coefficient_label(power)
        x_term = "x" if power == 1 else f"x^{{{power}}}"
        polynomial_terms.append(f"{coefficient_label}{x_term}")
        coefficient_rows.append(
            f"| ${coefficient_label}$ | "
            f"{format_parenthetical_uncertainty(coefficient, coefficient_error)} |"
        )
    polynomial_formula = "+".join(polynomial_terms)
    result_rows = [
        f"| $p_c$ | {format_parenthetical_uncertainty(pc, pc_error)} |",
        f"| $p_c$ (%) | {format_parenthetical_uncertainty(100 * pc, 100 * pc_error)} |",
        f"| $\\nu$ | {format_parenthetical_uncertainty(nu, nu_error)} |",
        *coefficient_rows,
        f"| $\\chi^2 / \\mathrm{{dof}}$ | {fit.chi2:.2f} / {fit.dof} = "
        f"{fit.chi2_per_dof:.2f} |",
        f"| goodness-of-fit $p$-value | {fit.p_value:.3g} |",
    ]
    display(
        Markdown(
            f"#### Finite-size scaling fit: {fit.label}\n\n"
            f"Degree-{fit.polynomial_degree} model: $P_L(p)={polynomial_formula}$, "
            f"with $x=(p-p_c)L^{{1/\\nu}}$. "
            f"The fit uses $p \\in [{p_min:.3f}, {p_max:.3f}]$ and "
            f"$L \\in {included_L}$.\n\n"
            "| Quantity | Fit result |\n|---|---:|\n"
            + "\n".join(result_rows)
        )
    )
    if fit.p_value < 0.05:
        display(
            Markdown(
                f"> **Fit-quality warning:** the degree-{fit.polynomial_degree} scaling model "
                "is rejected at the 5% level for this window. The covariance errors are "
                "statistical only; vary the degree, fit window, and minimum $L$ before quoting "
                "a final uncertainty."
            )
        )

    _, ax = plt.subplots(figsize=(5, 5))
    for cur_l in included_L:
        l_mask = fit.L_fit == cur_l
        x = (fit.p_fit[l_mask] - pc) * cur_l ** (1 / nu)
        order = np.argsort(x)
        ax.errorbar(
            x[order],
            fit.rate_fit[l_mask][order],
            yerr=fit.rate_error_fit[l_mask][order],
            fmt="o",
            capsize=3,
            label=f"L = {cur_l}",
        )
    x_line = np.linspace(
        np.min((fit.p_fit - pc) * fit.L_fit ** (1 / nu)),
        np.max((fit.p_fit - pc) * fit.L_fit ** (1 / nu)),
        300,
    )
    y_line = A + sum(
        coefficient * x_line**power
        for power, coefficient in enumerate(coefficients, 1)
    )
    ax.plot(
        x_line,
        y_line,
        "k--",
        label=f"degree-{fit.polynomial_degree} fit",
    )
    ax.set_xlabel(r"$(p-p_c)L^{1/\nu}$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Scaling collapse: {fit.label}")
    ax.legend()
    plt.show()


def display_fit_summary(
    fits: Sequence[FiniteSizeScalingFit], *, title: str = "Fit summary"
) -> None:
    """Display threshold and fit-quality results for several fits."""

    rows = [
        "| Timing and block | degree | $p_c$ | $p_c$ (%) | $\\nu$ | "
        "$\\chi^2 / \\mathrm{dof}$ | fit $p$-value |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in fits:
        rows.append(
            f"| {fit.label} | {fit.polynomial_degree} "
            f"| {format_parenthetical_uncertainty(fit.pc, fit.pc_error)} "
            f"| {format_parenthetical_uncertainty(100 * fit.pc, 100 * fit.pc_error)} "
            f"| {format_parenthetical_uncertainty(fit.nu, fit.nu_error)} "
            f"| {fit.chi2_per_dof:.2f} | {fit.p_value:.3g} |"
        )
    display(Markdown(f"## {title}\n\n" + "\n".join(rows)))


def _scaling_ansatz_model(
    pL: tuple[np.ndarray, np.ndarray],
    *parameters: float,
    polynomial_order: int,
    include_finite_size_correction: bool,
) -> np.ndarray:
    p, L = pL
    pc, nu, A = parameters[:3]
    coefficients = parameters[3 : 3 + polynomial_order]
    x = (p - pc) * L ** (1 / nu)
    result = A + sum(
        coefficient * x**order
        for order, coefficient in enumerate(coefficients, 1)
    )
    if include_finite_size_correction:
        E, mu = parameters[-2:]
        result = result + E * L ** (-1 / mu)
    return result


def fit_scaling_ansatz(
    scan: FailureScan,
    *,
    polynomial_order: int,
    include_finite_size_correction: bool = False,
    rate_name: str = "overall",
) -> ScalingAnsatzFit:
    """Run one of the exploratory scaling fits used by the older notebooks."""

    polynomial_order = _validate_polynomial_degree(polynomial_order)
    p, L, rate, rate_error = scan.fit_arrays(rate_name)

    def model(pL, *parameters):
        return _scaling_ansatz_model(
            pL,
            *parameters,
            polynomial_order=polynomial_order,
            include_finite_size_correction=include_finite_size_correction,
        )

    parameter_count = 3 + polynomial_order
    fit_kwargs = {"p0": [1.0] * parameter_count}
    if include_finite_size_correction:
        parameter_count = 3 + polynomial_order + 2
        fit_kwargs = {
            "p0": [0.2, 1.5, 0.25] + [0.0] * polynomial_order + [0.0, 1.0],
            "bounds": (
                [-np.inf] * (parameter_count - 1) + [0.0],
                [np.inf] * (parameter_count - 1) + [100.0],
            ),
        }

    parameters, covariance, info, message, status = curve_fit(
        model,
        (p, L),
        rate,
        full_output=True,
        sigma=rate_error,
        absolute_sigma=True,
        maxfev=100000,
        **fit_kwargs,
    )
    return ScalingAnsatzFit(
        polynomial_order=polynomial_order,
        include_finite_size_correction=include_finite_size_correction,
        parameters=parameters,
        covariance=covariance,
        weighted_absolute_residual=float(np.sum(np.abs(info["fvec"]))),
        solver_message=message,
        solver_status=status,
    )


def print_scaling_ansatz_fit(fit: ScalingAnsatzFit) -> None:
    """Print the results shown by the original exploratory fit cells."""

    print("Fitted parameters:", fit.parameters)
    print("Parameter standard deviations:", fit.parameter_errors)
    print("Weighted absolute residual:", fit.weighted_absolute_residual)
