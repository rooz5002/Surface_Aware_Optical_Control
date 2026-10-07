#!/usr/bin/env python3
'''
surface_aware_optic_test_v3.py

=================================

Reproducible analysis of the GapCal laser-triangulation experiment.

Input:
    CSV or Excel (.xlsx/.xls) file with exactly these required columns:

    method,surface,lighting,repeat,reference_mm,measured_mm,
    brightness,dynamic_range,laser,green,exposure_us

The program does NOT generate or modify measurements.

This program reads the CSV log produced by GapCal and calculates the measurement
to produce test result analysis for our academic paper
in 
Robotics and AI Labs
Huaian Univversity

By: Dr. Amir Ali Mokhtarzadeh
*****************************

The experiments using Gapcal, a contactless gap and flush measurment
device developed in our Lab 2019-2021

Experimental structure:
    3 methods
    6 surfaces
    2 lighting conditions
    10 repeats per method/surface/lighting configuration
    360 measurements total

The calculations follow the definitions stated in the accompanying paper:
    error = measured - reference
    MAE   = mean(abs(error))
    RMSE  = sqrt(mean(error^2))
    bias  = mean(error)
    sigma = sample standard deviation of error (ddof=1)

A configuration is classified as in-tolerance when:
    abs(mean error) <= 0.10 mm
    AND at least 8 of 10 repeats have abs(error) <= 0.10 mm.

Usage:
    python analyze_gapcal_test.py Gapcal_Test_log.xlsx
    python analyze_gapcal_test.py Gapcal_Test_log.csv --output results
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Experimental constants from the paper
# ---------------------------------------------------------------------------

TOLERANCE_MM = 0.10
MIN_TOLERANCE_REPEATS = 8
EXPECTED_REPEATS = 10

EXPECTED_METHODS = [
    "Fixed baseline",
    "k-NN + refinement",
    "SLM variant",
]

EXPECTED_SURFACES = [
    "White",
    "Black",
    "Grey",
    "Red",
    "Blue",
    "Green",
]

EXPECTED_LIGHTING = [
    "Fluorescent",
    "Natural",
]

CORE_COLUMNS = [
    "method", "surface", "lighting", "repeat",
    "reference_mm", "measured_mm", "brightness",
    "dynamic_range", "laser", "green", "exposure_us",
]

OPTIONAL_METADATA_COLUMNS = [
    "timestamp", "run_id", "run_number",
    "software_version", "git_commit",
    "simulation_active", "demo_mode",
]

NUMERIC_COLUMNS = [
    "repeat", "reference_mm", "measured_mm", "brightness",
    "dynamic_range", "laser", "green", "exposure_us",
]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_bias(errors: pd.Series | np.ndarray) -> float:
    x = np.asarray(errors, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot calculate bias from an empty dataset.")
    return float(np.mean(x))


def calculate_mae(errors: pd.Series | np.ndarray) -> float:
    x = np.asarray(errors, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot calculate MAE from an empty dataset.")
    return float(np.mean(np.abs(x)))


def calculate_rmse(errors: pd.Series | np.ndarray) -> float:
    x = np.asarray(errors, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot calculate RMSE from an empty dataset.")
    return float(np.sqrt(np.mean(np.square(x))))


def calculate_sigma(errors: pd.Series | np.ndarray) -> float:
    x = np.asarray(errors, dtype=float)
    if x.size < 2:
        raise ValueError("At least two measurements are required for sigma.")
    return float(np.std(x, ddof=1))


def tolerance_count(errors: pd.Series | np.ndarray) -> int:
    x = np.asarray(errors, dtype=float)
    return int(np.sum(np.abs(x) <= TOLERANCE_MM))


def configuration_in_tolerance(errors: pd.Series | np.ndarray) -> bool:
    x = np.asarray(errors, dtype=float)

    if x.size != EXPECTED_REPEATS:
        raise ValueError(
            f"Configuration contains {x.size} repeats; "
            f"expected exactly {EXPECTED_REPEATS}."
        )

    return (
        abs(calculate_bias(x)) <= TOLERANCE_MM
        and tolerance_count(x) >= MIN_TOLERANCE_REPEATS
    )


def summarize_errors(errors: pd.Series | np.ndarray) -> dict:
    return {
        "N": int(len(errors)),
        "Bias_mm": calculate_bias(errors),
        "MAE_mm": calculate_mae(errors),
        "RMSE_mm": calculate_rmse(errors),
        "Sigma_mm": calculate_sigma(errors),
        "Within_Tolerance_Count": tolerance_count(errors),
        "Within_Tolerance_Percent": (
            100.0 * tolerance_count(errors) / len(errors)
        ),
    }


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            "Use .csv, .xlsx or .xls."
        )

    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    extra = [c for c in df.columns if c not in CORE_COLUMNS + OPTIONAL_METADATA_COLUMNS]
    if extra:
        print("WARNING: extra columns found and ignored: " + ", ".join(extra))

    keep = CORE_COLUMNS + [c for c in OPTIONAL_METADATA_COLUMNS if c in df.columns]
    data = df[keep].copy()

    methods = set(data["method"].dropna().astype(str))
    surfaces = set(data["surface"].dropna().astype(str))
    lighting = set(data["lighting"].dropna().astype(str))
    if methods != set(EXPECTED_METHODS):
        raise ValueError(f"Method values do not match the experimental design. Expected: {EXPECTED_METHODS}; Found: {sorted(methods)}")
    if surfaces != set(EXPECTED_SURFACES):
        raise ValueError(f"Surface values do not match the experimental design. Expected: {EXPECTED_SURFACES}; Found: {sorted(surfaces)}")
    if lighting != set(EXPECTED_LIGHTING):
        raise ValueError(f"Lighting values do not match the experimental design. Expected: {EXPECTED_LIGHTING}; Found: {sorted(lighting)}")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            rows = (data.index[data[column].isna()] + 2).tolist()
            raise ValueError(f"Invalid or missing numeric values in '{column}' (spreadsheet rows: {rows[:10]}).")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite numeric values found in '{column}'.")

    for column in ["method", "surface", "lighting"]:
        if data[column].isna().any() or (data[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Blank values found in '{column}'.")

    if len(data) < 360:
        raise ValueError(f"Expected at least 360 measurements for the standard experiment, found {len(data)}.")

    # Each ordinary configuration must have 10 repeats. Extra runs are allowed.
    group_columns = ["method", "surface", "lighting"]
    counts = data.groupby(group_columns, sort=False).size()
    expected_groups = 36
    if len(counts) != expected_groups:
        raise ValueError(f"Expected {expected_groups} method/surface/lighting configurations, found {len(counts)}.")

    # Use run_id when supplied. Otherwise create one run per configuration.
    if "run_id" not in data.columns:
        data["run_id"] = (
            data["method"].astype(str) + "|" +
            data["surface"].astype(str) + "|" +
            data["lighting"].astype(str)
        )
    else:
        if data["run_id"].isna().any() or (data["run_id"].astype(str).str.strip() == "").any():
            raise ValueError("Blank values found in 'run_id'.")
        data["run_id"] = data["run_id"].astype(str)

    if "run_number" not in data.columns:
        data["run_number"] = 1
    else:
        data["run_number"] = pd.to_numeric(data["run_number"], errors="coerce")
        if data["run_number"].isna().any():
            raise ValueError("Invalid run_number values found.")

    run_counts = data.groupby("run_id", sort=False).size()
    bad_runs = run_counts[run_counts != EXPECTED_REPEATS]
    if not bad_runs.empty:
        raise ValueError("Every run_id must contain exactly 10 repeats:\n" + "\n".join(f"{idx}: {count} rows" for idx, count in bad_runs.items()))

    duplicate_rows = data.duplicated(subset=["run_id", "repeat"], keep=False)
    if duplicate_rows.any():
        raise ValueError("Duplicate repeat numbers found within run_id:\n" + data.loc[duplicate_rows, ["run_id", "repeat"]].to_string(index=False))

    invalid_repeats = data[~data["repeat"].isin(range(1, EXPECTED_REPEATS + 1))]
    if not invalid_repeats.empty:
        raise ValueError("Repeat numbers must be 1 through 10.")

    reference_values = data["reference_mm"].unique()
    if len(reference_values) != 1 or not np.isclose(reference_values[0], 1.000, atol=1e-12):
        raise ValueError(f"The dataset reference_mm is not consistently 1.000 mm. Found values: {reference_values}")

    # Timestamp is optional for backward compatibility, but if present it is validated.
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        if data["timestamp"].isna().any():
            raise ValueError("Invalid timestamp values found.")
        if not data["timestamp"].is_unique:
            raise ValueError("Timestamp values must be unique for individual measurements.")

    # Provenance fields are retained. True simulation/demo flags are reported, not rejected.
    data["error_mm"] = data["measured_mm"] - data["reference_mm"]
    return data


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------

def overall_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in EXPECTED_METHODS:
        subset = data[data["method"] == method]
        stats = summarize_errors(subset["error_mm"])

        rows.append({
            "Method": method,
            **stats,
        })

    return pd.DataFrame(rows)


def per_configuration_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, subset in data.groupby("run_id", sort=False):
        errors = subset["error_mm"].to_numpy()
        first = subset.iloc[0]
        rows.append({
            "Run_ID": run_id,
            "Run_Number": int(first["run_number"]),
            "Method": first["method"],
            "Surface": first["surface"],
            "Lighting": first["lighting"],
            "N": len(errors),
            "Mean_Error_mm": calculate_bias(errors),
            "MAE_mm": calculate_mae(errors),
            "RMSE_mm": calculate_rmse(errors),
            "Sigma_mm": calculate_sigma(errors),
            "Within_Tolerance_Count": tolerance_count(errors),
            "Within_Tolerance_Percent": 100.0*tolerance_count(errors)/len(errors),
            "Configuration_In_Tolerance": configuration_in_tolerance(errors),
        })
    return pd.DataFrame(rows)


def per_surface_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method in EXPECTED_METHODS:
        for surface in EXPECTED_SURFACES:
            subset = data[
                (data["method"] == method)
                & (data["surface"] == surface)
            ]

            stats = summarize_errors(subset["error_mm"])

            rows.append({
                "Method": method,
                "Surface": surface,
                **stats,
            })

    return pd.DataFrame(rows)


def table_iv(data: pd.DataFrame) -> pd.DataFrame:
    config = per_configuration_statistics(data)
    overall = overall_statistics(data)

    rows = []

    for method in EXPECTED_METHODS:
        method_config = config[config["Method"] == method]

        # The paper's tolerance criterion is configuration-level.
        passed = int(
            method_config["Configuration_In_Tolerance"].sum()
        )
        total = len(method_config)
        percentage = 100.0 * passed / total

        overall_row = overall[overall["Method"] == method].iloc[0]

        rows.append({
            "Configuration": method,
            "In_Tolerance_Configurations": passed,
            "Total_Configurations": total,
            "In_Tolerance_Percent": percentage,
            "RMSE_mm": overall_row["RMSE_mm"],
            "Sigma_mm": overall_row["Sigma_mm"],
            "Iteration": np.nan,
            "Latency_s": np.nan,
        })

    return pd.DataFrame(rows)


def run_comparison(data: pd.DataFrame) -> pd.DataFrame:
    config = per_configuration_statistics(data)
    counts = config.groupby(["Method", "Surface", "Lighting"]).size()
    repeated = counts[counts > 1].index
    if len(repeated) == 0:
        return pd.DataFrame()
    return config.merge(
        config.groupby(["Method", "Surface", "Lighting"])["Sigma_mm"]
        .agg(["min", "max", "mean"]).reset_index(),
        on=["Method", "Surface", "Lighting"],
        suffixes=("", "_Across_Runs")
    )[config["Method"].isin(config["Method"])]


def provenance_table(data: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["software_version", "git_commit", "simulation_active", "demo_mode"] if c in data.columns]
    if not cols:
        return pd.DataFrame({"Information": ["No provenance metadata columns were supplied."]})
    rows = []
    for c in cols:
        rows.append({"Field": c, "Value": ", ".join(sorted(data[c].astype(str).unique()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(
    data: pd.DataFrame,
    per_surface: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Error distributions
    plt.figure(figsize=(10, 6))
    for method in EXPECTED_METHODS:
        errors = data.loc[data["method"] == method, "error_mm"]
        plt.hist(errors, bins=25, alpha=0.55, label=method)

    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axvline(TOLERANCE_MM, linestyle=":", linewidth=1)
    plt.axvline(-TOLERANCE_MM, linestyle=":", linewidth=1)
    plt.xlabel("Measurement error (mm)")
    plt.ylabel("Frequency")
    plt.title("GapCal Measurement Error Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "error_distributions.png", dpi=300)
    plt.close()

    # 2. RMSE by surface
    plt.figure(figsize=(11, 6))

    x = np.arange(len(EXPECTED_SURFACES))
    width = 0.25

    for i, method in enumerate(EXPECTED_METHODS):
        values = []
        for surface in EXPECTED_SURFACES:
            row = per_surface[
                (per_surface["Method"] == method)
                & (per_surface["Surface"] == surface)
            ]
            values.append(float(row["RMSE_mm"].iloc[0]))

        plt.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
        )

    plt.xticks(x, EXPECTED_SURFACES)
    plt.ylabel("RMSE (mm)")
    plt.xlabel("Surface")
    plt.title("RMSE by Surface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "rmse_by_surface.png", dpi=300)
    plt.close()

    # 3. Repeatability sigma by surface
    plt.figure(figsize=(11, 6))

    for i, method in enumerate(EXPECTED_METHODS):
        values = []
        for surface in EXPECTED_SURFACES:
            row = per_surface[
                (per_surface["Method"] == method)
                & (per_surface["Surface"] == surface)
            ]
            values.append(float(row["Sigma_mm"].iloc[0]))

        plt.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method,
        )

    plt.xticks(x, EXPECTED_SURFACES)
    plt.ylabel("Repeatability σ (mm)")
    plt.xlabel("Surface")
    plt.title("Repeatability by Surface")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "repeatability_by_surface.png", dpi=300)
    plt.close()

    # 4. Measured values by method
    plt.figure(figsize=(10, 6))

    for method in EXPECTED_METHODS:
        values = data.loc[
            data["method"] == method, "measured_mm"
        ].to_numpy()

        plt.plot(
            np.arange(1, len(values) + 1),
            values,
            ".",
            alpha=0.5,
            label=method,
        )

    plt.axhline(1.000, linestyle="--", linewidth=1, label="Reference")
    plt.xlabel("Measurement index")
    plt.ylabel("Measured gap (mm)")
    plt.title("GapCal Measurements")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "measured_values.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_summary(
    data: pd.DataFrame,
    overall: pd.DataFrame,
    table: pd.DataFrame,
) -> None:
    print()
    print("=" * 72)
    print("surface aware optic test v3")
    print("Robotics annd AI Research Labs")
    print("Huai'an University")
    print("Under Dr. Amir Ali Mokhtarrzadeh Supervision")
    print("=" * 72)
    print(f"Measurements:       {len(data)}")
    print(f"Independent runs:   {data['run_id'].nunique()}")
    print(f"Methods:            {data['method'].nunique()}")
    print(f"Surfaces:           {data['surface'].nunique()}")
    print(f"Lighting conditions:{data['lighting'].nunique()}")
    print(f"Reference:          {data['reference_mm'].iloc[0]:.3f} mm")
    print(f"Repeats/config:     {EXPECTED_REPEATS}")
    print()

    display = overall[
        [
            "Method",
            "N",
            "Bias_mm",
            "MAE_mm",
            "RMSE_mm",
            "Sigma_mm",
            "Within_Tolerance_Percent",
        ]
    ].copy()

    print("OVERALL STATISTICS")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    print("TABLE IV TOLERANCE RESULT")
    print(
        table[
            [
                "Configuration",
                "In_Tolerance_Configurations",
                "Total_Configurations",
                "In_Tolerance_Percent",
                "RMSE_mm",
                "Sigma_mm",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print(
        "Tolerance criterion: mean error within ±0.10 mm AND "
        "at least 8 of 10 repeats within ±0.10 mm."
    )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyse GapCal experimental measurements."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="GapCal test log (.csv, .xlsx or .xls)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results)",
    )

    args = parser.parse_args()

    try:
        raw = load_input(args.input)
        data = validate_dataset(raw)

        args.output.mkdir(parents=True, exist_ok=True)

        overall = overall_statistics(data)
        configuration = per_configuration_statistics(data)
        surface = per_surface_statistics(data)
        table = table_iv(data)

        # Save derived tables.
        overall.to_csv(
            args.output / "Overall_Statistics.csv",
            index=False,
        )
        configuration.to_csv(
            args.output / "Per_Configuration_Results.csv",
            index=False,
        )
        surface.to_csv(
            args.output / "Per_Surface_Results.csv",
            index=False,
        )
        table.to_csv(
            args.output / "Table_IV.csv",
            index=False,
        )

        runs = run_comparison(data)
        if not runs.empty:
            runs.to_csv(args.output / "Run_Comparison.csv", index=False)
        provenance_table(data).to_csv(args.output / "Provenance.csv", index=False)

        make_figures(data, surface, args.output)
        print_summary(data, overall, table)

        print()
        print(f"Results written to: {args.output.resolve()}")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
