"""Thesis-facing descriptive analysis of calibrated corpus predictions.

The analysis intentionally treats the predictions as fixed model outputs.  It does not turn
model-assisted labels into human-valid prevalence estimates. Generated intervals cluster by the
canonical Reddit submission thread recovered through an exact source-corpus join.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ANALYSIS_VERSION = "1.1.0"
DECOMPOSITION_START_YEAR = 2020
DECOMPOSITION_END_YEAR = 2025
DEFAULT_INPUT = Path(
    "data/private-hf-modernbert-probability-random-corpus-source-v2/data/*.parquet"
)
DEFAULT_MANIFEST = Path(
    "data/private-hf-modernbert-probability-random-corpus-source-v2/manifest.json"
)
DEFAULT_INVENTORY = Path(
    "data/private-hf-modernbert-probability-random-corpus-source-v2/"
    "provenance/canonical-inventory.json"
)
DEFAULT_THREAD_MAPPING = Path("data/private-corpus-thread-mapping-v1/thread-mapping.parquet")
DEFAULT_THREAD_MAPPING_RECEIPT = Path("data/private-corpus-thread-mapping-v1/receipt.json")
DEFAULT_OUTPUT = Path("outputs/corpus-analysis-v1")


@dataclass(frozen=True)
class Target:
    slug: str
    label: str


TARGETS = (
    Target("china_general", "China generally"),
    Target("government_ccp", "Government / CCP"),
    Target("people_identity", "People / identity"),
    Target("culture_media", "Culture / media"),
    Target("company_tech_product", "Companies / technology"),
)
TARGET_LABELS = {target.slug: target.label for target in TARGETS}

SUBREDDIT_STRATA = {
    "China": "domain",
    "ChineseLanguage": "domain",
    "Sino": "domain",
    "geopolitics": "news_political",
    "news": "news_political",
    "worldnews": "news_political",
    "AskReddit": "general_interest",
    "funny": "general_interest",
    "gaming": "general_interest",
    "todayilearned": "general_interest",
}
STRATUM_LABELS = {
    "domain": "China-focused communities",
    "news_political": "News / politics",
    "general_interest": "General interest",
}

COLOURS = {
    "navy": "#17324D",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#737B85",
    "light_grey": "#D8DEE6",
    "ink": "#17212B",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_value(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _query_dicts(connection: Any, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _sql_identifier_list(columns: Sequence[str]) -> str:
    allowed = {
        "month",
        "year",
        "subreddit",
        "subreddit_stratum",
        "content_type",
        "retrieval_mode",
        "target",
    }
    if not set(columns).issubset(allowed):
        raise ValueError(f"unsupported grouping columns: {columns}")
    return ", ".join(columns)


def _ratio_query(group_columns: Sequence[str], *, hard: bool = False) -> str:
    grouping = _sql_identifier_list(group_columns)
    prefix = f"{grouping}, " if grouping else ""
    group_by = f"{grouping}, thread_id" if grouping else "thread_id"
    outer_group = f"GROUP BY {grouping}" if grouping else ""
    outer_order = f"ORDER BY {grouping}" if grouping else ""

    if hard:
        row_filter = "WHERE relevance = 'material' AND target_present"
        numerator = "SUM(CASE stance WHEN 'positive' THEN 1 WHEN 'negative' THEN -1 ELSE 0 END)"
        denominator = "COUNT(*)"
        positive = "SUM(CASE WHEN stance = 'positive' THEN 1 ELSE 0 END)"
        negative = "SUM(CASE WHEN stance = 'negative' THEN 1 ELSE 0 END)"
        neutral = "SUM(CASE WHEN stance = 'no_directed_stance' THEN 1 ELSE 0 END)"
    else:
        row_filter = ""
        numerator = "SUM(weight * (positive_probability - negative_probability))"
        denominator = "SUM(weight)"
        positive = "SUM(weight * positive_probability)"
        negative = "SUM(weight * negative_probability)"
        neutral = "SUM(weight * neutral_probability)"

    return f"""
        WITH per_cluster AS (
            SELECT
                {prefix}thread_id,
                {numerator}::DOUBLE AS numerator,
                {denominator}::DOUBLE AS denominator,
                {positive}::DOUBLE AS positive_mass,
                {negative}::DOUBLE AS negative_mass,
                {neutral}::DOUBLE AS neutral_mass
            FROM stance_long
            {row_filter}
            GROUP BY {group_by}
        ), statistics AS (
            SELECT
                {prefix}
                COUNT(*)::BIGINT AS n_clusters,
                SUM(numerator)::DOUBLE AS sum_numerator,
                SUM(denominator)::DOUBLE AS sum_denominator,
                SUM(numerator * numerator)::DOUBLE AS sum_numerator_sq,
                SUM(numerator * denominator)::DOUBLE AS sum_numerator_denominator,
                SUM(denominator * denominator)::DOUBLE AS sum_denominator_sq,
                SUM(positive_mass)::DOUBLE AS positive_mass,
                SUM(negative_mass)::DOUBLE AS negative_mass,
                SUM(neutral_mass)::DOUBLE AS neutral_mass
            FROM per_cluster
            {outer_group}
        )
        SELECT
            {prefix}
            n_clusters,
            sum_denominator AS target_weight,
            sum_numerator / NULLIF(sum_denominator, 0) AS score,
            SQRT(
                GREATEST(
                    0,
                    n_clusters::DOUBLE / NULLIF(n_clusters - 1, 0)
                    * (
                        sum_numerator_sq
                        - 2 * (sum_numerator / NULLIF(sum_denominator, 0))
                            * sum_numerator_denominator
                        + POWER(sum_numerator / NULLIF(sum_denominator, 0), 2)
                            * sum_denominator_sq
                    )
                    / NULLIF(POWER(sum_denominator, 2), 0)
                )
            ) AS thread_cluster_se,
            positive_mass / NULLIF(sum_denominator, 0) AS positive_share,
            negative_mass / NULLIF(sum_denominator, 0) AS negative_share,
            neutral_mass / NULLIF(sum_denominator, 0) AS neutral_share
        FROM statistics
        {outer_order}
    """


def _target_union_sql() -> str:
    statements: list[str] = []
    for target in TARGETS:
        slug = target.slug
        statements.append(
            f"""
            SELECT
                corpus_position, thread_id, month, year, subreddit, subreddit_stratum,
                content_type, retrieval_mode,
                relevance_probability, relevance,
                '{slug}' AS target,
                target_{slug}_probability AS target_probability,
                target_{slug}_present AS target_present,
                stance_{slug}_positive_probability AS positive_probability,
                stance_{slug}_negative_probability AS negative_probability,
                stance_{slug}_no_directed_stance_probability AS neutral_probability,
                stance_{slug} AS stance,
                relevance_probability * target_{slug}_probability AS weight
            FROM base
            """
        )
    return " UNION ALL ".join(statements)


def prepare_connection(input_glob: Path, thread_mapping_path: Path) -> Any:
    import duckdb

    connection = duckdb.connect(database=":memory:")
    connection.execute("SET TimeZone='UTC'")
    connection.execute("SET threads=4")
    escaped_glob = str(input_glob).replace("'", "''")
    escaped_mapping = str(thread_mapping_path).replace("'", "''")
    connection.execute(
        f"""
        CREATE TEMP VIEW predictions AS
        SELECT
            *,
            strftime(created_utc AT TIME ZONE 'UTC', '%Y-%m-01') AS month
        FROM read_parquet('{escaped_glob}')
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW base AS
        SELECT
            p.*,
            m.thread_id,
            CASE
                WHEN lower(p.subreddit) IN ('china', 'chineselanguage', 'sino') THEN 'domain'
                WHEN lower(p.subreddit) IN ('geopolitics', 'news', 'worldnews')
                    THEN 'news_political'
                WHEN lower(p.subreddit) IN ('askreddit', 'funny', 'gaming', 'todayilearned')
                    THEN 'general_interest'
                ELSE NULL
            END AS subreddit_stratum
        FROM predictions AS p
        LEFT JOIN read_parquet('{escaped_mapping}') AS m USING (corpus_position)
        WHERE NOT p.calibration_member
        """
    )
    connection.execute(f"CREATE TEMP TABLE stance_long AS {_target_union_sql()}")
    return connection


def validate_source(
    connection: Any,
    manifest: Mapping[str, Any],
    thread_mapping_path: Path,
    thread_mapping_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_body = {
        key: value for key, value in thread_mapping_receipt.items() if key != "receipt_id"
    }
    descriptor = thread_mapping_receipt.get("mapping")
    if (
        thread_mapping_receipt.get("receipt_id") != canonical_sha256(receipt_body)
        or thread_mapping_receipt.get("status") != "complete"
        or not isinstance(descriptor, Mapping)
        or descriptor.get("sha256") != sha256_file(thread_mapping_path)
    ):
        raise RuntimeError("thread-mapping receipt or payload binding failed")
    checks = _query_dicts(
        connection,
        """
        SELECT
            (SELECT COUNT(*) FROM base) AS base_rows,
            (SELECT COUNT(*) FROM predictions) AS prediction_rows,
            (SELECT COUNT(*) FROM stance_long) AS long_rows,
            (SELECT COUNT(DISTINCT thread_id) FROM base) AS thread_clusters,
            (SELECT COUNT(*) FROM base WHERE thread_id IS NULL OR thread_id = '')
                AS missing_thread_ids,
            (SELECT COUNT(*) FROM base WHERE subreddit_stratum IS NULL)
                AS missing_subreddit_strata,
            (SELECT MAX(records) FROM (
                SELECT thread_id, COUNT(*) AS records FROM base GROUP BY thread_id
            )) AS maximum_records_per_thread,
            (SELECT COUNT(DISTINCT month) FROM base) AS months,
            (SELECT MIN(month) FROM base) AS first_month,
            (SELECT MAX(month) FROM base) AS last_month,
            (SELECT MIN(year) FROM base) AS first_year,
            (SELECT MAX(year) FROM base) AS last_year,
            (SELECT COUNT(*) FROM stance_long
             WHERE weight < 0 OR weight > 1 OR NOT isfinite(weight)) AS invalid_weights,
            (SELECT MAX(ABS(
                positive_probability + negative_probability + neutral_probability - 1
             )) FROM stance_long) AS max_stance_probability_sum_error
        """,
    )[0]
    observed_strata = {
        str(row["subreddit"]): str(row["subreddit_stratum"])
        for row in _query_dicts(
            connection,
            "SELECT DISTINCT subreddit, subreddit_stratum FROM base ORDER BY subreddit",
        )
    }
    expected_rows = int(manifest["default_label_unseen_rows"])
    expected_predictions = int(manifest["corpus_rows"])
    expected_long = expected_rows * len(TARGETS)
    failures: list[str] = []
    if checks["base_rows"] != expected_rows:
        failures.append(f"base row count {checks['base_rows']} != {expected_rows}")
    if checks["prediction_rows"] != expected_predictions:
        failures.append(
            f"prediction row count {checks['prediction_rows']} != {expected_predictions}"
        )
    if checks["long_rows"] != expected_long:
        failures.append(f"long row count {checks['long_rows']} != {expected_long}")
    if checks["months"] != 72 or checks["first_month"] != "2020-01-01":
        failures.append("monthly coverage is not the expected 2020-01 through 2025-12 span")
    if checks["last_month"] != "2025-12-01":
        failures.append(f"last UTC month is {checks['last_month']}, expected 2025-12-01")
    if checks["invalid_weights"]:
        failures.append(f"found {checks['invalid_weights']} invalid weights")
    if checks["missing_thread_ids"] or checks["thread_clusters"] != expected_rows:
        failures.append(
            "thread mapping is missing or does not preserve one thread per analysis row"
        )
    if checks["missing_subreddit_strata"]:
        failures.append(f"found {checks['missing_subreddit_strata']} rows without a stratum")
    if observed_strata != SUBREDDIT_STRATA:
        failures.append(
            f"subreddit-stratum mapping differs from the canonical mapping: {observed_strata}"
        )
    if checks["maximum_records_per_thread"] != 1:
        failures.append(
            "analysis corpus unexpectedly contains multiple retained records per thread"
        )
    if checks["max_stance_probability_sum_error"] > 1e-6:
        failures.append("stance probabilities do not sum to one within tolerance")
    if failures:
        raise RuntimeError("source validation failed: " + "; ".join(failures))
    return checks


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["subreddit"]), str(row["content_type"]), str(row["target"]))


def standardise_cells(
    cell_year: Sequence[Mapping[str, Any]],
    *,
    start_year: int = DECOMPOSITION_START_YEAR,
    end_year: int = DECOMPOSITION_END_YEAR,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by_year: dict[int, dict[tuple[str, str, str], Mapping[str, Any]]] = defaultdict(dict)
    pooled_num: dict[tuple[str, str, str], float] = defaultdict(float)
    pooled_den: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in cell_year:
        year = int(row["year"])
        key = _cell_key(row)
        by_year[year][key] = row
        denominator = float(row["target_weight"])
        pooled_den[key] += denominator
        pooled_num[key] += denominator * float(row["score"])

    keys = sorted(pooled_den)
    total_pooled_den = sum(pooled_den.values())
    reference_share = {key: pooled_den[key] / total_pooled_den for key in keys}
    reference_mean = {key: pooled_num[key] / pooled_den[key] for key in keys}

    standardised: list[dict[str, Any]] = []
    for year in sorted(by_year):
        rows = by_year[year]
        year_den = sum(float(row["target_weight"]) for row in rows.values())
        observed = (
            sum(float(row["target_weight"]) * float(row["score"]) for row in rows.values())
            / year_den
        )
        available_ref = sum(reference_share[key] for key in rows)
        fixed_composition = (
            sum(reference_share[key] * float(rows[key]["score"]) for key in rows) / available_ref
        )
        composition_only = sum(
            float(rows[key]["target_weight"]) / year_den * reference_mean[key] for key in rows
        )
        standardised.append(
            {
                "year": year,
                "observed_score": observed,
                "fixed_composition_score": fixed_composition,
                "composition_only_score": composition_only,
                "cells": len(rows),
            }
        )

    start = by_year[start_year]
    end = by_year[end_year]
    shared = sorted(set(start) & set(end))
    start_total = sum(float(row["target_weight"]) for row in start.values())
    end_total = sum(float(row["target_weight"]) for row in end.values())
    contributions: list[dict[str, Any]] = []
    within_total = 0.0
    composition_total = 0.0
    for key in shared:
        start_share = float(start[key]["target_weight"]) / start_total
        end_share = float(end[key]["target_weight"]) / end_total
        start_mean = float(start[key]["score"])
        end_mean = float(end[key]["score"])
        within = 0.5 * (start_share + end_share) * (end_mean - start_mean)
        composition = 0.5 * (start_mean + end_mean) * (end_share - start_share)
        within_total += within
        composition_total += composition
        contributions.append(
            {
                "subreddit": key[0],
                "content_type": key[1],
                "target": key[2],
                "start_share": start_share,
                "end_share": end_share,
                "start_score": start_mean,
                "end_score": end_mean,
                "within_contribution": within,
                "composition_contribution": composition,
                "total_contribution": within + composition,
            }
        )
    total = within_total + composition_total
    decomposition = {
        "start_year": start_year,
        "end_year": end_year,
        "start_score": sum(
            float(row["target_weight"]) * float(row["score"]) for row in start.values()
        )
        / start_total,
        "end_score": sum(float(row["target_weight"]) * float(row["score"]) for row in end.values())
        / end_total,
        "total_change": total,
        "within_change": within_total,
        "composition_change": composition_total,
        "within_share": within_total / total,
        "composition_share": composition_total / total,
        "shared_cells": len(shared),
    }
    return standardised, decomposition, contributions


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def linear_trend(rows: Sequence[Mapping[str, Any]], value: str = "score") -> dict[str, float]:
    ys = [float(row[value]) for row in rows]
    xs = list(range(len(ys)))
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    ss_x = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / ss_x
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    ss_total = sum((y - y_mean) ** 2 for y in ys)
    ss_residual = sum((y - fit) ** 2 for y, fit in zip(ys, fitted, strict=True))
    return {
        "slope_per_month": slope,
        "slope_per_year": slope * 12,
        "r_squared": 1 - ss_residual / ss_total if ss_total else 0.0,
    }


def source_volume(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str, str], int] = defaultdict(int)
    for entry in inventory["entries"]:
        for partition in entry["partitions"]:
            relative = str(partition["relative_path"])
            pieces = {
                piece.split("=", 1)[0]: piece.split("=", 1)[1]
                for piece in relative.split("/")
                if "=" in piece
            }
            key = (int(pieces["year"]), pieces["subreddit"], pieces["content_type"])
            counts[key] += int(partition["rows"])
    return [
        {
            "year": year,
            "subreddit": subreddit,
            "content_type": content_type,
            "source_rows": rows,
        }
        for (year, subreddit, content_type), rows in sorted(counts.items())
    ]


def _monthly_dates(rows: Sequence[Mapping[str, Any]]) -> list[datetime]:
    return [datetime.strptime(str(row["month"]), "%Y-%m-%d") for row in rows]


def _moving_average(values: Sequence[float], window: int = 3) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(_mean(values[start : index + 1]))
    return result


def _figure_setup() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLOURS["ink"],
            "axes.labelcolor": COLOURS["ink"],
            "axes.titlecolor": COLOURS["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": COLOURS["ink"],
            "ytick.color": COLOURS["ink"],
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    return plt


def _clean_axis(axis: Any, *, grid: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid, color=COLOURS["light_grey"], linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save_figure(figure: Any, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        ("png", {"dpi": 220}),
        ("pdf", {}),
        ("svg", {}),
    ):
        figure.savefig(figure_dir / f"{name}.{suffix}", **options)


def plot_overview(monthly: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _figure_setup()
    dates = _monthly_dates(monthly)
    values = [float(row["score"]) for row in monthly]
    standard_errors = [float(row["thread_cluster_se"]) for row in monthly]
    trailing = _moving_average(values)
    lower = [value - 1.96 * se for value, se in zip(values, standard_errors, strict=True)]
    upper = [value + 1.96 * se for value, se in zip(values, standard_errors, strict=True)]

    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.fill_between(dates, lower, upper, color=COLOURS["sky"], alpha=0.18, linewidth=0)
    axis.plot(dates, values, color=COLOURS["sky"], linewidth=1.2, alpha=0.85)
    axis.plot(dates, trailing, color=COLOURS["navy"], linewidth=2.6)
    axis.axhline(0, color=COLOURS["ink"], linewidth=1)
    axis.set_ylim(min(lower) - 0.02, 0.02)
    axis.set_title("Predicted stance became less negative after 2022", loc="left", pad=30)
    axis.text(
        0,
        1.01,
        "Monthly expected stance; dark line is the trailing three-month mean",
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    axis.set_ylabel("Expected stance  (-1 negative, +1 positive)")
    axis.set_xlabel("")
    _clean_axis(axis)
    axis.text(
        1,
        -0.18,
        "Shading: 95% thread-cluster interval; model uncertainty excluded",
        transform=axis.transAxes,
        ha="right",
        color=COLOURS["grey"],
        fontsize=8,
    )
    _save_figure(figure, figure_dir, "fig01_monthly_overview")
    plt.close(figure)


def plot_stance_components(annual: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _figure_setup()
    years = [int(row["year"]) for row in annual]
    figure, axis = plt.subplots(figsize=(7.7, 4.4), constrained_layout=True)
    series = (
        ("negative_share", "Negative", COLOURS["red"]),
        ("neutral_share", "No directed stance", COLOURS["grey"]),
        ("positive_share", "Positive", COLOURS["green"]),
    )
    for field, label, colour in series:
        axis.plot(
            years,
            [float(row[field]) for row in annual],
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=label,
            color=colour,
        )
    axis.set_ylim(0, 0.70)
    axis.set_xticks(years)
    axis.set_ylabel("Share of target-weighted stance mass")
    axis.set_title(
        "The change is mainly negative stance becoming non-directional", loc="left", pad=30
    )
    axis.text(
        0,
        1.01,
        "Positive stance rises only modestly",
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    axis.legend(ncol=3, loc="upper center")
    _clean_axis(axis)
    _save_figure(figure, figure_dir, "fig02_stance_components")
    plt.close(figure)


def plot_target_trends(target_year: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _figure_setup()
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in target_year:
        by_target[str(row["target"])].append(row)
    figure, axes = plt.subplots(3, 2, figsize=(8.0, 8.0), sharex=True)
    axes_flat = list(axes.flat)
    for axis, target in zip(axes_flat, TARGETS, strict=False):
        rows = sorted(by_target[target.slug], key=lambda item: int(item["year"]))
        years = [int(row["year"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        errors = [1.96 * float(row["thread_cluster_se"]) for row in rows]
        axis.errorbar(
            years,
            scores,
            yerr=errors,
            color=COLOURS["blue"],
            marker="o",
            linewidth=2,
            capsize=2,
        )
        axis.axhline(0, color=COLOURS["ink"], linewidth=0.9)
        axis.set_title(target.label, loc="left", fontsize=10)
        axis.set_xticks(years)
        axis.set_ylim(-0.43, 0.16)
        _clean_axis(axis)
    axes_flat[-1].axis("off")
    figure.suptitle(
        "Target matters: one aggregate combines different constructs",
        x=0.02,
        y=0.99,
        ha="left",
    )
    figure.text(
        0.02,
        0.955,
        "Annual expected stance with 95% thread-cluster intervals; model uncertainty excluded",
        color=COLOURS["grey"],
        va="top",
    )
    figure.supylabel("Expected stance")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.86, hspace=0.32, wspace=0.16)
    _save_figure(figure, figure_dir, "fig03_target_trends")
    plt.close(figure)


def _heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_field: str,
    column_field: str,
    value_field: str,
    row_order: Sequence[Any],
    column_order: Sequence[Any],
    title: str,
    subtitle: str,
    path_name: str,
    figure_dir: Path,
) -> None:
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    plt = _figure_setup()
    lookup = {(row[row_field], row[column_field]): float(row[value_field]) for row in rows}
    matrix = np.array(
        [
            [lookup.get((row_value, column_value), np.nan) for column_value in column_order]
            for row_value in row_order
        ]
    )
    figure, axis = plt.subplots(
        figsize=(8.0, max(4.0, 0.43 * len(row_order) + 1.5)), constrained_layout=True
    )
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-0.42, vcenter=0, vmax=0.20),
    )
    axis.set_xticks(range(len(column_order)), [str(item) for item in column_order])
    axis.set_yticks(range(len(row_order)), [str(item) for item in row_order])
    axis.set_title(title, loc="left", pad=30)
    axis.text(
        0,
        1.01,
        subtitle,
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    for y_index in range(len(row_order)):
        for x_index in range(len(column_order)):
            value = matrix[y_index, x_index]
            if math.isnan(value):
                continue
            colour = "white" if value < -0.24 or value > 0.25 else COLOURS["ink"]
            axis.text(
                x_index,
                y_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
            )
    colourbar = figure.colorbar(image, ax=axis, shrink=0.75, pad=0.02)
    colourbar.set_label("Expected stance")
    for spine in axis.spines.values():
        spine.set_visible(False)
    _save_figure(figure, figure_dir, path_name)
    plt.close(figure)


def plot_subreddit_year_heatmap(
    subreddit_year: Sequence[Mapping[str, Any]], figure_dir: Path
) -> None:
    latest = {
        str(row["subreddit"]): float(row["score"])
        for row in subreddit_year
        if int(row["year"]) == 2025
    }
    order = sorted(latest, key=latest.get)
    _heatmap(
        subreddit_year,
        row_field="subreddit",
        column_field="year",
        value_field="score",
        row_order=order,
        column_order=range(2020, 2026),
        title="Community differences exceed the platform-wide trend",
        subtitle="Annual expected stance by subreddit",
        path_name="fig04_subreddit_trends",
        figure_dir=figure_dir,
    )


def plot_composition(
    standardised: Sequence[Mapping[str, Any]],
    decomposition: Mapping[str, Any],
    figure_dir: Path,
) -> None:
    plt = _figure_setup()
    years = [int(row["year"]) for row in standardised]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(9.0, 4.3), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    for field, label, colour, marker in (
        ("observed_score", "Observed mix", COLOURS["navy"], "o"),
        ("fixed_composition_score", "Fixed composition", COLOURS["blue"], "s"),
        ("composition_only_score", "Composition only", COLOURS["orange"], "^"),
    ):
        left.plot(
            years,
            [float(row[field]) for row in standardised],
            marker=marker,
            linewidth=2,
            color=colour,
            label=label,
        )
    left.axhline(0, color=COLOURS["ink"], linewidth=0.9)
    left.set_xticks(years)
    left.set_ylabel("Expected stance")
    left.legend(loc="lower right")
    _clean_axis(left)

    values = [float(decomposition["composition_change"]), float(decomposition["within_change"])]
    labels = ["Changing mix", "Within comparable cells"]
    colours = [COLOURS["orange"], COLOURS["blue"]]
    bars = right.barh(labels, values, color=colours, height=0.55)
    right.axvline(0, color=COLOURS["ink"], linewidth=0.9)
    for bar, value in zip(bars, values, strict=True):
        share = value / float(decomposition["total_change"])
        right.text(
            value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}  ({share:.0%})",
            va="center",
        )
    right.set_xlim(0, max(values) * 1.5)
    right.set_xlabel(
        f"Contribution to {decomposition['start_year']}-{decomposition['end_year']} change"
    )
    _clean_axis(right, grid="x")
    figure.suptitle(
        "Both composition and within-group movement explain the change",
        x=0.02,
        y=0.98,
        ha="left",
    )
    figure.text(
        0.02,
        0.91,
        "Cells are subreddit x content type x target; symmetric decomposition",
        color=COLOURS["grey"],
        va="top",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.80, wspace=0.48)
    _save_figure(figure, figure_dir, "fig05_composition_decomposition")
    plt.close(figure)


def plot_content_type(content_year: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _figure_setup()
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in content_year:
        by_type[str(row["content_type"])].append(row)
    figure, axis = plt.subplots(figsize=(7.7, 4.2), constrained_layout=True)
    for content_type, colour in (("comment", COLOURS["red"]), ("submission", COLOURS["green"])):
        rows = sorted(by_type[content_type], key=lambda item: int(item["year"]))
        axis.plot(
            [int(row["year"]) for row in rows],
            [float(row["score"]) for row in rows],
            marker="o",
            linewidth=2.3,
            label=content_type.title(),
            color=colour,
        )
    axis.axhline(0, color=COLOURS["ink"], linewidth=0.9)
    axis.set_xticks(range(2020, 2026))
    axis.set_ylabel("Expected stance")
    axis.set_title("Comments are consistently more negative than submissions", loc="left", pad=30)
    axis.text(
        0,
        1.01,
        "Annual probability-weighted estimates",
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    axis.legend()
    _clean_axis(axis)
    _save_figure(figure, figure_dir, "fig06_content_type")
    plt.close(figure)


def plot_volume(
    candidate_year: Sequence[Mapping[str, Any]],
    source_year: Sequence[Mapping[str, Any]],
    figure_dir: Path,
) -> None:
    plt = _figure_setup()
    candidates = {int(row["year"]): int(row["candidate_rows"]) for row in candidate_year}
    sources = {int(row["year"]): int(row["source_rows"]) for row in source_year}
    years = sorted(candidates)
    rates = [candidates[year] / sources[year] * 100_000 for year in years]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.8, 6.2), sharex=True, constrained_layout=True
    )
    top.plot(
        years,
        [sources[year] / sources[years[0]] * 100 for year in years],
        marker="o",
        color=COLOURS["grey"],
        linewidth=2.2,
        label="All source rows",
    )
    top.plot(
        years,
        [candidates[year] / candidates[years[0]] * 100 for year in years],
        marker="s",
        color=COLOURS["blue"],
        linewidth=2.2,
        label="Candidate rows",
    )
    top.set_ylabel("Volume index  (2020 = 100)")
    top.legend(ncol=2)
    _clean_axis(top)
    bottom.plot(years, rates, marker="o", color=COLOURS["orange"], linewidth=2.2)
    bottom.set_ylabel("Candidates per 100,000 source rows")
    bottom.set_xticks(years)
    _clean_axis(bottom)
    figure.suptitle("Fewer candidates mostly coincide with lower source volume", x=0.02, ha="left")
    figure.text(
        0.02,
        0.965,
        "Absolute counts decline, while the candidate rate recovers after 2022",
        color=COLOURS["grey"],
        va="top",
    )
    _save_figure(figure, figure_dir, "fig07_volume_and_coverage")
    plt.close(figure)


def plot_subreddit_target(subreddit_target: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    subreddit_pooled: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
    for row in subreddit_target:
        numerator, denominator = subreddit_pooled[str(row["subreddit"])]
        weight = float(row["target_weight"])
        subreddit_pooled[str(row["subreddit"])] = (
            numerator + weight * float(row["score"]),
            denominator + weight,
        )
    order = sorted(
        subreddit_pooled, key=lambda key: subreddit_pooled[key][0] / subreddit_pooled[key][1]
    )
    plot_labels = {
        "china_general": "China\ngenerally",
        "government_ccp": "Government /\nCCP",
        "people_identity": "People /\nidentity",
        "culture_media": "Culture /\nmedia",
        "company_tech_product": "Companies /\ntechnology",
    }
    display_rows = [dict(row, target=plot_labels[str(row["target"])]) for row in subreddit_target]
    _heatmap(
        display_rows,
        row_field="subreddit",
        column_field="target",
        value_field="score",
        row_order=order,
        column_order=[plot_labels[target.slug] for target in TARGETS],
        title="The aggregate masks subreddit x target structure",
        subtitle="Pooled 2020-2025 expected stance",
        path_name="fig08_subreddit_target",
        figure_dir=figure_dir,
    )


def plot_sensitivity(
    annual: Sequence[Mapping[str, Any]],
    hard_annual: Sequence[Mapping[str, Any]],
    standardised: Sequence[Mapping[str, Any]],
    equal_community: Sequence[Mapping[str, Any]],
    figure_dir: Path,
) -> None:
    plt = _figure_setup()
    years = [int(row["year"]) for row in annual]
    hard = {int(row["year"]): float(row["score"]) for row in hard_annual}
    fixed = {int(row["year"]): float(row["fixed_composition_score"]) for row in standardised}
    equal = {int(row["year"]): float(row["score"]) for row in equal_community}
    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    series = (
        ("Probability-weighted", [float(row["score"]) for row in annual], COLOURS["navy"], "o"),
        ("Hard labels", [hard[year] for year in years], COLOURS["red"], "^"),
        ("Fixed composition", [fixed[year] for year in years], COLOURS["blue"], "s"),
        ("Equal community", [equal[year] for year in years], COLOURS["orange"], "D"),
    )
    for label, values, colour, marker in series:
        axis.plot(years, values, label=label, color=colour, marker=marker, linewidth=2)
    axis.axhline(0, color=COLOURS["ink"], linewidth=0.9)
    axis.set_xticks(years)
    axis.set_ylabel("Expected stance")
    axis.set_title("The direction is robust; the level depends on aggregation", loc="left", pad=30)
    axis.text(
        0,
        1.01,
        "Four defensible summaries of the same model predictions",
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    axis.legend(ncol=2)
    _clean_axis(axis)
    _save_figure(figure, figure_dir, "fig09_aggregation_sensitivity")
    plt.close(figure)


def plot_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], annual: Sequence[Mapping[str, Any]], figure_dir: Path
) -> None:
    plt = _figure_setup()
    years = [int(row["year"]) for row in diagnostics]
    neutral = {int(row["year"]): float(row["neutral_share"]) for row in annual}
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 4.2))
    left.plot(
        years,
        [float(row["mean_relevance_probability"]) for row in diagnostics],
        marker="o",
        linewidth=2.2,
        color=COLOURS["blue"],
        label="Mean probability",
    )
    left.plot(
        years,
        [float(row["hard_material_share"]) for row in diagnostics],
        marker="s",
        linewidth=2.2,
        color=COLOURS["orange"],
        label="Hard material share",
    )
    left.set_ylim(0.70, 0.90)
    left.set_xticks(years)
    left.set_title("Relevance", loc="left", fontsize=11)
    left.legend()
    _clean_axis(left)
    right.plot(
        years, [neutral[year] for year in years], marker="o", linewidth=2.2, color=COLOURS["grey"]
    )
    right.set_ylim(0.50, 0.68)
    right.set_xticks(years)
    right.set_title("No directed stance", loc="left", fontsize=11)
    _clean_axis(right)
    figure.suptitle("Measurement outputs also shift over time", x=0.02, y=0.98, ha="left")
    figure.text(
        0.02,
        0.91,
        "These diagnostics motivate human validation by year and subgroup",
        color=COLOURS["grey"],
        va="top",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.79, wspace=0.28)
    _save_figure(figure, figure_dir, "fig10_measurement_diagnostics")
    plt.close(figure)


def plot_stratum_trends(stratum_year: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _figure_setup()
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in stratum_year:
        by_stratum[str(row["subreddit_stratum"])].append(row)
    styles = {
        "domain": (COLOURS["purple"], "o"),
        "news_political": (COLOURS["blue"], "s"),
        "general_interest": (COLOURS["orange"], "^"),
    }
    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for stratum in ("domain", "news_political", "general_interest"):
        rows = sorted(by_stratum[stratum], key=lambda item: int(item["year"]))
        colour, marker = styles[stratum]
        axis.plot(
            [int(row["year"]) for row in rows],
            [float(row["score"]) for row in rows],
            label=STRATUM_LABELS[stratum],
            color=colour,
            marker=marker,
            linewidth=2.2,
        )
    axis.axhline(0, color=COLOURS["ink"], linewidth=0.9)
    axis.set_xticks(range(2020, 2026))
    axis.set_ylabel("Expected stance")
    axis.set_title("The broad shift appears in all three subreddit strata", loc="left", pad=30)
    axis.text(
        0,
        1.01,
        "China-focused, news/political and general-interest communities remain distinct",
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    axis.legend(ncol=3, fontsize=8.5)
    _clean_axis(axis)
    _save_figure(figure, figure_dir, "fig11_stratum_trends")
    plt.close(figure)


def plot_retrieval_modes(retrieval_year: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _figure_setup()
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in retrieval_year:
        by_mode[str(row["retrieval_mode"])].append(row)
    figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    for mode, label, colour, marker in (
        ("direct", "Direct lexical matches", COLOURS["navy"], "o"),
        ("expanded_only", "Semantic expansion only", COLOURS["green"], "s"),
    ):
        rows = sorted(by_mode[mode], key=lambda item: int(item["year"]))
        axis.plot(
            [int(row["year"]) for row in rows],
            [float(row["score"]) for row in rows],
            label=label,
            color=colour,
            marker=marker,
            linewidth=2.2,
        )
    axis.axhline(0, color=COLOURS["ink"], linewidth=0.9)
    axis.set_xticks(range(2020, 2026))
    axis.set_ylabel("Expected stance")
    axis.set_title(
        "The temporal direction is not unique to one retrieval route", loc="left", pad=30
    )
    axis.text(
        0,
        1.01,
        "Direct and semantic-expansion candidates converge by 2024-2025",
        transform=axis.transAxes,
        color=COLOURS["grey"],
        va="bottom",
    )
    axis.legend(ncol=2)
    _clean_axis(axis)
    _save_figure(figure, figure_dir, "fig12_retrieval_robustness")
    plt.close(figure)


def _sum_contributions(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"within_contribution": 0.0, "composition_contribution": 0.0}
    )
    for row in rows:
        key = str(row[field])
        totals[key]["within_contribution"] += float(row["within_contribution"])
        totals[key]["composition_contribution"] += float(row["composition_contribution"])
    return [
        {
            field: key,
            **values,
            "total_contribution": values["within_contribution"]
            + values["composition_contribution"],
        }
        for key, values in totals.items()
    ]


def plot_decomposition_drivers(
    contributions: Sequence[Mapping[str, Any]], figure_dir: Path
) -> None:
    plt = _figure_setup()
    by_target = _sum_contributions(contributions, "target")
    by_subreddit = sorted(
        _sum_contributions(contributions, "subreddit"),
        key=lambda row: float(row["total_contribution"]),
    )
    target_lookup = {str(row["target"]): row for row in by_target}
    ordered_targets = [target_lookup[target.slug] for target in TARGETS]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8.6, 7.4))

    for axis, rows, field, title in (
        (top, ordered_targets, "target", "Contribution by target"),
        (bottom, by_subreddit, "subreddit", "Contribution by subreddit"),
    ):
        labels = [
            TARGET_LABELS[str(row[field])] if field == "target" else f"r/{row[field]}"
            for row in rows
        ]
        positions = list(range(len(rows)))
        height = 0.36
        axis.barh(
            [position - height / 2 for position in positions],
            [float(row["within_contribution"]) for row in rows],
            height=height,
            color=COLOURS["blue"],
            label="Within comparable cells",
        )
        axis.barh(
            [position + height / 2 for position in positions],
            [float(row["composition_contribution"]) for row in rows],
            height=height,
            color=COLOURS["orange"],
            label="Changing mix",
        )
        axis.axvline(0, color=COLOURS["ink"], linewidth=0.9)
        axis.set_yticks(positions, labels)
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_xlabel(
            f"Contribution to {DECOMPOSITION_START_YEAR}-{DECOMPOSITION_END_YEAR} change"
        )
        _clean_axis(axis, grid="x")
    top.legend(ncol=2, loc="lower right")
    figure.suptitle(
        "The aggregate change has identifiable target and community drivers",
        x=0.02,
        y=0.99,
        ha="left",
    )
    figure.text(
        0.02,
        0.955,
        "Positive values make the corpus less negative; symmetric decomposition",
        color=COLOURS["grey"],
        va="top",
    )
    figure.subplots_adjust(left=0.22, right=0.98, bottom=0.08, top=0.87, hspace=0.48)
    _save_figure(figure, figure_dir, "fig13_decomposition_drivers")
    plt.close(figure)


def _group_weighted_rows(
    rows: Sequence[Mapping[str, Any]], group: str, *, value_field: str = "score"
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[group]].append(row)
    result: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        denominator = sum(float(item["target_weight"]) for item in items)
        score = (
            sum(float(item["target_weight"]) * float(item[value_field]) for item in items)
            / denominator
        )
        result.append({group: key, "score": score, "target_weight": denominator})
    return result


def equal_community_by_year(subreddit_year: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in subreddit_year:
        grouped[int(row["year"])].append(float(row["score"]))
    return [
        {"year": year, "score": _mean(values), "communities": len(values)}
        for year, values in sorted(grouped.items())
    ]


def _aggregate_source_year(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[int, int] = defaultdict(int)
    for row in rows:
        totals[int(row["year"])] += int(row["source_rows"])
    return [{"year": year, "source_rows": value} for year, value in sorted(totals.items())]


def _aggregate_candidate_year(connection: Any) -> list[dict[str, Any]]:
    return _query_dicts(
        connection,
        "SELECT year, COUNT(*)::BIGINT AS candidate_rows FROM base GROUP BY year ORDER BY year",
    )


def _diagnostics(connection: Any) -> list[dict[str, Any]]:
    return _query_dicts(
        connection,
        """
        SELECT
            year,
            COUNT(*)::BIGINT AS rows,
            AVG(relevance_probability)::DOUBLE AS mean_relevance_probability,
            AVG(CASE WHEN relevance = 'material' THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS hard_material_share
        FROM base
        GROUP BY year
        ORDER BY year
        """,
    )


def _report(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    annual: Sequence[Mapping[str, Any]],
    target_year: Sequence[Mapping[str, Any]],
    subreddit_year: Sequence[Mapping[str, Any]],
    stratum_year: Sequence[Mapping[str, Any]],
    content_year: Sequence[Mapping[str, Any]],
    retrieval_year: Sequence[Mapping[str, Any]],
    standardised: Sequence[Mapping[str, Any]],
    decomposition: Mapping[str, Any],
    contributions: Sequence[Mapping[str, Any]],
    monthly_trend: Mapping[str, float],
    candidate_year: Sequence[Mapping[str, Any]],
    source_year: Sequence[Mapping[str, Any]],
) -> None:
    annual_lookup = {int(row["year"]): row for row in annual}
    start = annual_lookup[2022]
    end = annual_lookup[2025]
    target_lookup = {(str(row["target"]), int(row["year"])): row for row in target_year}
    subreddit_lookup = {(str(row["subreddit"]), int(row["year"])): row for row in subreddit_year}
    stratum_lookup = {
        (str(row["subreddit_stratum"]), int(row["year"])): row for row in stratum_year
    }
    content_lookup = {(str(row["content_type"]), int(row["year"])): row for row in content_year}
    retrieval_lookup = {
        (str(row["retrieval_mode"]), int(row["year"])): row for row in retrieval_year
    }
    candidate_lookup = {int(row["year"]): int(row["candidate_rows"]) for row in candidate_year}
    source_lookup = {int(row["year"]): int(row["source_rows"]) for row in source_year}

    target_lines = []
    for target in TARGETS:
        first = float(target_lookup[(target.slug, 2022)]["score"])
        last = float(target_lookup[(target.slug, 2025)]["score"])
        target_lines.append(
            f"- **{target.label}:** {first:+.3f} → {last:+.3f} ({last - first:+.3f})."
        )

    subreddit_changes = []
    subreddits = sorted({str(row["subreddit"]) for row in subreddit_year})
    for subreddit in subreddits:
        first = float(subreddit_lookup[(subreddit, 2022)]["score"])
        last = float(subreddit_lookup[(subreddit, 2025)]["score"])
        subreddit_changes.append((last - first, subreddit, first, last))
    subreddit_lines = [
        f"- **r/{subreddit}:** {first:+.3f} → {last:+.3f} ({change:+.3f})."
        for change, subreddit, first, last in sorted(subreddit_changes, reverse=True)
    ]
    stratum_lines = []
    for stratum in ("domain", "news_political", "general_interest"):
        first = float(stratum_lookup[(stratum, 2022)]["score"])
        last = float(stratum_lookup[(stratum, 2025)]["score"])
        stratum_lines.append(
            f"- **{STRATUM_LABELS[stratum]}:** {first:+.3f} → {last:+.3f} ({last - first:+.3f})."
        )
    target_drivers = sorted(
        _sum_contributions(contributions, "target"),
        key=lambda row: abs(float(row["total_contribution"])),
        reverse=True,
    )
    subreddit_drivers = sorted(
        _sum_contributions(contributions, "subreddit"),
        key=lambda row: abs(float(row["total_contribution"])),
        reverse=True,
    )
    driver_lines = [
        f"- Largest target driver: **{TARGET_LABELS[str(target_drivers[0]['target'])]}** "
        f"({float(target_drivers[0]['total_contribution']):+.3f}).",
        f"- Largest community driver: **r/{subreddit_drivers[0]['subreddit']}** "
        f"({float(subreddit_drivers[0]['total_contribution']):+.3f}).",
    ]

    report = f"""# Reddit China stance corpus analysis

- **Version:** {ANALYSIS_VERSION}
- **Corpus revision:** `{manifest["export_id"]}`
- **Default scope:** {validation["base_rows"]:,} label-unseen candidate records,
January 2020-December 2025 UTC

## Executive answer

The model-predicted aggregate is negative in every year, reaches its most negative annual value in
2022 ({float(start["score"]):+.3f}), and becomes substantially less negative by 2025
({float(end["score"]):+.3f}). This is not a flat line: the 2022-2025 change is
{float(end["score"]) - float(start["score"]):+.3f}, and the monthly linear trend over the full
period
is {monthly_trend["slope_per_year"]:+.3f} points per year (descriptive R² =
{monthly_trend["r_squared"]:.2f}). It is also not a transition to broadly positive stance.

Most of the movement is negative model probability becoming `no_directed_stance`: from 2022 to
2025, positive mass changes by {float(end["positive_share"]) - float(start["positive_share"]):+.3f},
negative mass by {float(end["negative_share"]) - float(start["negative_share"]):+.3f}, and
non-directional mass by {float(end["neutral_share"]) - float(start["neutral_share"]):+.3f}.

The headline average is substantively incomplete. A symmetric decomposition attributes
{float(decomposition["composition_share"]):.0%} of the
{int(decomposition["start_year"])}-{int(decomposition["end_year"])} movement to the changing mix
of subreddits, content types and targets, and
{float(decomposition["within_share"]):.0%} to movement within comparable cells.

![Monthly overview](figures/fig01_monthly_overview.png)

## RQ1 — How did target-specific expressed stance change over time?

The central descriptive result is a 2022 trough followed by a sustained move towards less-negative
predictions. “Less negative” is the correct wording: positive mass changes little, while negative
mass falls and non-directional mass rises.

![Stance components](figures/fig02_stance_components.png)

Target-specific results show that there is no single China stance construct:

{chr(10).join(target_lines)}

![Target trends](figures/fig03_target_trends.png)

## RQ2 — How do trends differ, and how much is composition?

The subreddit contrasts are larger than the aggregate movement. The 2022-2025 changes are:

{chr(10).join(subreddit_lines)}

![Subreddit trends](figures/fig04_subreddit_trends.png)

The same direction is visible in all three pre-defined subreddit strata, although their levels and
effect sizes differ:

{chr(10).join(stratum_lines)}

![Subreddit-stratum trends](figures/fig11_stratum_trends.png)

The decomposition uses 100 cells (10 subreddits x 2 content types x 5 targets). The observed
change is {float(decomposition["total_change"]):+.3f}: composition contributes
{float(decomposition["composition_change"]):+.3f}, while within-cell movement contributes
{float(decomposition["within_change"]):+.3f}. This means neither “the audience changed” nor “stance
changed everywhere” is an adequate standalone explanation.

![Composition decomposition](figures/fig05_composition_decomposition.png)

The decomposition can also be traced to specific targets and communities:

{chr(10).join(driver_lines)}

![Decomposition drivers](figures/fig13_decomposition_drivers.png)

Comments remain more negative than submissions. In 2022 the estimates are
{float(content_lookup[("comment", 2022)]["score"]):+.3f} and
{float(content_lookup[("submission", 2022)]["score"]):+.3f}; in 2025 they are
{float(content_lookup[("comment", 2025)]["score"]):+.3f} and
{float(content_lookup[("submission", 2025)]["score"]):+.3f}.

![Content type](figures/fig06_content_type.png)

Candidate volume falls from {candidate_lookup[2020]:,} rows in 2020 to {candidate_lookup[2025]:,}
in 2025, but source volume also falls from {source_lookup[2020]:,} to {source_lookup[2025]:,} rows.
The candidate rate is {candidate_lookup[2020] / source_lookup[2020] * 100_000:.1f} per 100,000
source rows in 2020 and {candidate_lookup[2025] / source_lookup[2025] * 100_000:.1f} in 2025.
The count decline alone is therefore not evidence of declining China discussion.

![Volume and coverage](figures/fig07_volume_and_coverage.png)

The retrieval-route check also points in the same temporal direction. Direct lexical matches move
from {float(retrieval_lookup[("direct", 2022)]["score"]):+.3f} in 2022 to
{float(retrieval_lookup[("direct", 2025)]["score"]):+.3f} in 2025; semantic-expansion-only
candidates move from {float(retrieval_lookup[("expanded_only", 2022)]["score"]):+.3f} to
{float(retrieval_lookup[("expanded_only", 2025)]["score"]):+.3f}. This makes a changing retrieval
route an unlikely standalone explanation for the trend.

![Retrieval-route robustness](figures/fig12_retrieval_robustness.png)

![Subreddit by target](figures/fig08_subreddit_target.png)

The broad temporal direction survives hard-label, equal-community and fixed-composition summaries,
but the absolute level changes. The thesis should report more than one estimand rather than treating
the volume-weighted platform aggregate as uniquely authoritative.

![Aggregation sensitivity](figures/fig09_aggregation_sensitivity.png)

## RQ3 — Were event deviations distinguishable from placebos?

**Not yet answerable as a confirmatory question.** The repository does not contain a frozen event
list with exact dates, scopes and windows. The vertical markers at January 2022 and January 2024 in
the exploratory draft are not labelled and have already been viewed against the outcomes. Testing
post-hoc selected dates as if they were pre-specified would be invalid.

For the final event analysis, freeze for every event: exact date, affected target(s), relevant
subreddit strata, pre/post window and matched-placebo rule. Report event estimates as
associations, not causal effects. A global all-target/all-community line is likely to dilute local
event responses.

## Measurement and uncertainty

![Measurement diagnostics](figures/fig10_measurement_diagnostics.png)

- These are calibrated ModernBERT predictions, not human annotations or population truth.
- The analysis joins every prediction to the canonical `submission_id`. The retained candidate
  corpus contains exactly one scored record for each of {validation["thread_clusters"]:,} threads,
  so record- and thread-clustered intervals coincide here.
- Intervals condition on the fitted model and exclude annotation uncertainty, calibration drift and
  model error. Those systematic uncertainties are likely more important than sampling error.
- The changing relevance and non-directional outputs make independent human validation by year,
  subreddit, target and predicted class essential before thesis-facing validity claims.
- The sample covers selected English-language China-related discourse in ten communities. It does
  not represent Reddit overall, Reddit users, China, or public opinion.

## Defensible thesis story

> In the selected English-language China-related Reddit candidate corpus, model-predicted expressed
> stance was negative throughout 2020-2025, reached a trough in 2022, and became less negative
> through 2025. The movement primarily reflects declining negative and increasing non-directional
> predictions rather than a broad rise in positive stance. Both changing subreddit/target
> composition and change within comparable discourse cells contribute, while substantial
> heterogeneity across communities and targets makes a single platform-wide average incomplete.

Avoid claiming that Reddit, Reddit users or public opinion “became more positive toward China.”
The data support a statement about model-predicted expressed stance in this selected corpus.

## Reproducibility

The complete numeric tables are under `tables/`; vector versions of every figure are under
`figures/`. `receipt.json` binds the source manifest, script and generated artefacts. Re-run with:

```bash
uv run --group analysis python -m reddit_china_stance.corpus_analysis_v1
```
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _write_receipt(
    output_dir: Path,
    *,
    manifest_path: Path,
    inventory_path: Path,
    thread_mapping_path: Path,
    thread_mapping_receipt_path: Path,
    validation: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "receipt.json":
            relative = str(path.relative_to(output_dir))
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    script_path = Path(__file__).resolve()
    receipt = {
        "kind": "reddit-china-stance-corpus-analysis-receipt-v1",
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_head": _git_value(["rev-parse", "HEAD"]),
        "git_status": _git_value(["status", "--short"]),
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "canonical_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
        "thread_mapping": {
            "path": str(thread_mapping_path),
            "sha256": sha256_file(thread_mapping_path),
            "receipt_path": str(thread_mapping_receipt_path),
            "receipt_sha256": sha256_file(thread_mapping_receipt_path),
        },
        "scope": {
            "exclude_calibration_members": True,
            "timezone": "UTC",
            "analytic_targets": [target.slug for target in TARGETS],
            "weighting": "relevance_probability * target_probability",
            "stance_score": "positive_probability - negative_probability",
            "uncertainty": "thread-clustered ratio SE; fitted-model uncertainty excluded",
        },
        "validation": dict(validation),
        "summary": dict(summary),
        "files": files,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_analysis(
    *,
    input_glob: Path,
    manifest_path: Path,
    inventory_path: Path,
    thread_mapping_path: Path,
    thread_mapping_receipt_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    thread_mapping_receipt = json.loads(thread_mapping_receipt_path.read_text(encoding="utf-8"))
    connection = prepare_connection(input_glob, thread_mapping_path)
    validation = validate_source(
        connection,
        manifest,
        thread_mapping_path,
        thread_mapping_receipt,
    )

    monthly = _query_dicts(connection, _ratio_query(["month"]))
    annual = _query_dicts(connection, _ratio_query(["year"]))
    target_year = _query_dicts(connection, _ratio_query(["target", "year"]))
    subreddit_year = _query_dicts(connection, _ratio_query(["subreddit", "year"]))
    stratum_year = _query_dicts(connection, _ratio_query(["subreddit_stratum", "year"]))
    content_year = _query_dicts(connection, _ratio_query(["content_type", "year"]))
    retrieval_year = _query_dicts(connection, _ratio_query(["retrieval_mode", "year"]))
    subreddit_target = _query_dicts(connection, _ratio_query(["subreddit", "target"]))
    cell_year = _query_dicts(
        connection, _ratio_query(["subreddit", "content_type", "target", "year"])
    )
    hard_annual = _query_dicts(connection, _ratio_query(["year"], hard=True))
    diagnostics = _diagnostics(connection)
    candidate_year = _aggregate_candidate_year(connection)
    source_detail = source_volume(inventory)
    source_year = _aggregate_source_year(source_detail)
    equal_community = equal_community_by_year(subreddit_year)
    standardised, decomposition, contributions = standardise_cells(cell_year)
    monthly_trend = linear_trend(monthly)
    decomposition_filename = (
        f"decomposition_cells_{DECOMPOSITION_START_YEAR}_{DECOMPOSITION_END_YEAR}.csv"
    )

    tables: dict[str, Sequence[Mapping[str, Any]]] = {
        "monthly_stance.csv": monthly,
        "annual_stance.csv": annual,
        "target_by_year.csv": target_year,
        "subreddit_by_year.csv": subreddit_year,
        "subreddit_stratum_by_year.csv": stratum_year,
        "content_type_by_year.csv": content_year,
        "retrieval_mode_by_year.csv": retrieval_year,
        "subreddit_by_target.csv": subreddit_target,
        "cell_by_year.csv": cell_year,
        "hard_label_annual.csv": hard_annual,
        "equal_community_annual.csv": equal_community,
        "standardised_annual.csv": standardised,
        decomposition_filename: contributions,
        "measurement_diagnostics.csv": diagnostics,
        "candidate_volume.csv": candidate_year,
        "canonical_source_volume.csv": source_detail,
        "canonical_source_volume_annual.csv": source_year,
    }
    for name, rows in tables.items():
        write_csv(table_dir / name, rows)

    plot_overview(monthly, figure_dir)
    plot_stance_components(annual, figure_dir)
    plot_target_trends(target_year, figure_dir)
    plot_subreddit_year_heatmap(subreddit_year, figure_dir)
    plot_composition(standardised, decomposition, figure_dir)
    plot_content_type(content_year, figure_dir)
    plot_volume(candidate_year, source_year, figure_dir)
    plot_subreddit_target(subreddit_target, figure_dir)
    plot_sensitivity(annual, hard_annual, standardised, equal_community, figure_dir)
    plot_diagnostics(diagnostics, annual, figure_dir)
    plot_stratum_trends(stratum_year, figure_dir)
    plot_retrieval_modes(retrieval_year, figure_dir)
    plot_decomposition_drivers(contributions, figure_dir)

    annual_lookup = {int(row["year"]): row for row in annual}
    summary = {
        "scope_rows": int(validation["base_rows"]),
        "monthly_trend": monthly_trend,
        "annual_score_2020": float(annual_lookup[2020]["score"]),
        "annual_score_2022": float(annual_lookup[2022]["score"]),
        "annual_score_2025": float(annual_lookup[2025]["score"]),
        "change_2020_2025": float(annual_lookup[2025]["score"])
        - float(annual_lookup[2020]["score"]),
        "change_2022_2025": float(annual_lookup[2025]["score"])
        - float(annual_lookup[2022]["score"]),
        "decomposition": decomposition,
        "event_analysis_status": "blocked_pending_frozen_event_definitions",
        "claim_boundary": "model-predicted expressed stance in the selected corpus",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _report(
        output_dir,
        manifest=manifest,
        validation=validation,
        annual=annual,
        target_year=target_year,
        subreddit_year=subreddit_year,
        stratum_year=stratum_year,
        content_year=content_year,
        retrieval_year=retrieval_year,
        standardised=standardised,
        decomposition=decomposition,
        contributions=contributions,
        monthly_trend=monthly_trend,
        candidate_year=candidate_year,
        source_year=source_year,
    )
    _write_receipt(
        output_dir,
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        thread_mapping_path=thread_mapping_path,
        thread_mapping_receipt_path=thread_mapping_receipt_path,
        validation=validation,
        summary=summary,
    )
    connection.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--thread-mapping", type=Path, default=DEFAULT_THREAD_MAPPING)
    parser.add_argument(
        "--thread-mapping-receipt", type=Path, default=DEFAULT_THREAD_MAPPING_RECEIPT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        input_glob=args.input,
        manifest_path=args.manifest,
        inventory_path=args.inventory,
        thread_mapping_path=args.thread_mapping,
        thread_mapping_receipt_path=args.thread_mapping_receipt,
        output_dir=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
