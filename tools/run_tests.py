#!/usr/bin/env python3

"""Serial native CTest runner for FlagDNN functional tests and benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "cmake" / "Operators.cmake"
HYGON_COMPARABLE_CASE_CATALOG = (
    ROOT
    / "backends"
    / "hygon"
    / "validation"
    / "benchmark"
    / "comparable_cases.json"
)
HYGON_SPEEDUP_METRIC = "hipdnn_median_us/flagdnn_median_us"
VALID_SUITES = ("functional", "benchmark")
BUILD_CONFIGURATION_FILE = ".flagdnn-build-config"
DEFAULT_OPERATORS = (
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "max",
    "min",
    "mod",
    "add_square",
    "cmp_eq",
)
PLATFORM_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
CTEST_SKIPPED_PATTERN = re.compile(
    r"(?:\*\*\*Skipped|\bNot Run\b.*\bSkipped\b|\(Skipped\))",
    flags=re.IGNORECASE,
)
CTEST_DISABLED_PATTERN = re.compile(
    r"(?:\*\*\*Not Run|\(Disabled\))",
    flags=re.IGNORECASE,
)
HYGON_VISIBILITY_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)
HYGON_CASE_FILTER_PATTERN = re.compile(r"^FLAGDNN_[A-Z0-9_]+_CASE$")
HYGON_CONVOLUTION_CASE_PATTERN = re.compile(
    r"^conv[123]d_(fprop|dgrad|wgrad)(?:_|$)"
)
HYGON_FUNCTIONAL_ACCOUNTING_MARKER_OVERRIDES = {
    "conv_fprop": "CONVOLUTION",
    "conv_dgrad": "CONVOLUTION",
    "conv_wgrad": "CONVOLUTION",
}
PROCESS_TERMINATION_GRACE_SECONDS = 5.0
PROCESS_KILL_GRACE_SECONDS = 5.0
PROCESS_CLEANUP_POLL_SECONDS = 0.05


def operator_manifests() -> dict[str, list[str]]:
    source = MANIFEST.read_text(encoding="utf-8")
    set_names = {
        "benchmark": "FLAGDNN_BENCHMARK_OPERATORS",
        "functional": "FLAGDNN_FUNCTIONAL_OPERATORS",
    }
    raw_sets: dict[str, list[str]] = {}
    for set_name in set_names.values():
        match = re.search(
            rf"set\({re.escape(set_name)}(?P<body>.*?)\)",
            source,
            flags=re.DOTALL,
        )
        if match is None:
            raise RuntimeError(
                f"Cannot parse {set_name} from operator manifest: {MANIFEST}"
            )
        body = re.sub(r"#.*", "", match.group("body"))
        raw_sets[set_name] = re.findall(
            r"\$\{[A-Z0-9_]+\}|[a-z][a-z0-9_]*", body
        )

    resolved: dict[str, list[str]] = {}

    def resolve(set_name: str, stack: tuple[str, ...] = ()) -> list[str]:
        if set_name in resolved:
            return resolved[set_name]
        if set_name in stack:
            raise RuntimeError(
                "Operator manifest contains a cyclic set expansion: "
                + " -> ".join((*stack, set_name))
            )
        if set_name not in raw_sets:
            raise RuntimeError(
                f"Operator manifest references unknown set {set_name}"
            )
        operators: list[str] = []
        for token in raw_sets[set_name]:
            if token.startswith("${"):
                operators.extend(resolve(token[2:-1], (*stack, set_name)))
            else:
                operators.append(token)
        if not operators or len(operators) != len(set(operators)):
            raise RuntimeError(
                f"{set_name} is empty or contains duplicate operators"
            )
        resolved[set_name] = operators
        return operators

    return {
        suite: list(resolve(set_name)) for suite, set_name in set_names.items()
    }


def requested_suites(value: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("--suites must select at least one suite")
    if raw == ["all"]:
        return list(VALID_SUITES)
    if "all" in raw:
        raise ValueError("--suites 'all' cannot be combined with other values")
    unsupported = set(raw) - set(VALID_SUITES)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported suite(s): {names}")
    return list(dict.fromkeys(raw))


def requested_operators(
    manifests: dict[str, list[str]],
    suites: list[str],
    value: str | None,
    list_file: Path | None,
) -> dict[str, list[str]]:
    if value is not None and list_file is not None:
        raise ValueError("--ops and --op-list-file are mutually exclusive")
    if value == "all":
        return {suite: list(manifests[suite]) for suite in suites}
    if value is not None:
        selected = [item.strip() for item in value.split(",") if item.strip()]
        if not selected:
            raise ValueError("--ops must select at least one operator")
    elif list_file is not None:
        selected = [
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not selected:
            raise ValueError(
                "--op-list-file must contain at least one operator"
            )
    else:
        selected = list(DEFAULT_OPERATORS)

    selected = list(dict.fromkeys(selected))
    unavailable = {
        suite: sorted(set(selected) - set(manifests[suite]))
        for suite in suites
    }
    unavailable = {
        suite: operators
        for suite, operators in unavailable.items()
        if operators
    }
    if unavailable:
        details = "; ".join(
            f"{suite}: {', '.join(operators)}"
            for suite, operators in unavailable.items()
        )
        raise ValueError(
            "Operators are not registered for every requested suite ("
            + details
            + ")"
        )
    return {suite: list(selected) for suite in suites}


def device_environment(
    platform: str,
    device: str | None,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    if platform == "hygon":
        for variable in tuple(environment):
            if HYGON_CASE_FILTER_PATTERN.fullmatch(variable) is not None:
                environment.pop(variable, None)
    if device is None:
        return environment
    if platform == "ascend":
        environment["ASCEND_RT_VISIBLE_DEVICES"] = device
        environment["NPU_VISIBLE_DEVICES"] = device
    elif platform == "hygon":
        for variable in HYGON_VISIBILITY_VARIABLES:
            environment.pop(variable, None)
        environment["HIP_VISIBLE_DEVICES"] = device
    else:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def resolve_build_directory(requested: Path | None, platform: str) -> Path:
    if requested is None:
        configured = os.environ.get("FLAGDNN_BUILD_DIR")
        requested = (
            Path(configured) if configured else Path("build") / platform
        )
    requested = requested.expanduser()
    if not requested.is_absolute():
        requested = ROOT / requested
    return requested.resolve()


def validate_build_directory(build_dir: Path) -> None:
    if (build_dir / "CTestTestfile.cmake").is_file():
        return

    installed_package = any(
        build_dir.glob("lib*/cmake/FlagDNN/FlagDNNConfig.cmake")
    )
    if installed_package or (build_dir / "share" / "flagdnn").is_dir():
        suggestion = "a configured build tree containing CTestTestfile.cmake"
        parent_build = build_dir.parent
        if (
            build_dir.name == "install"
            and (parent_build / "CTestTestfile.cmake").is_file()
        ):
            suggestion = str(parent_build)
        raise ValueError(
            f"{build_dir} is an installed FlagDNN SDK; validation tests are "
            f"not installed. Use --build-dir {suggestion}"
        )

    raise ValueError(
        f"{build_dir} is not a configured CTest build directory; "
        "run tools/build.sh with tests enabled or pass --build-dir"
    )


def resolve_build_configuration(
    build_dir: Path, requested: str | None
) -> str | None:
    """Resolve the CTest configuration for single- and multi-config builds."""

    def valid(value: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9_.+-]+", value) is not None

    if requested is not None:
        requested = requested.strip()
        if not requested or not valid(requested):
            raise ValueError(
                "--config must contain a CMake configuration name"
            )
        return requested

    cache_values: dict[str, str] = {}
    cache = build_dir / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("CMAKE_BUILD_TYPE:") or line.startswith(
                "CMAKE_CONFIGURATION_TYPES:"
            ):
                key, _, value = line.partition("=")
                cache_values[key.partition(":")[0]] = value.strip()
    build_type = cache_values.get("CMAKE_BUILD_TYPE", "")
    if build_type:
        return build_type
    marker = build_dir / BUILD_CONFIGURATION_FILE
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if not value or not valid(value):
            raise ValueError(f"invalid build configuration marker: {marker}")
        return value
    if cache_values.get("CMAKE_CONFIGURATION_TYPES"):
        return "Release"
    return None


def benchmark_records(
    output: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    required_keys = {
        "schema_version",
        "kind",
        "provider",
        "case",
        "unit",
        "median",
        "p90",
        "samples",
    }

    def positive_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 0
        )

    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(
                "malformed benchmark JSON "
                f"(line {error.lineno}, column {error.colno}): {line}"
            )
            continue
        if not isinstance(record, dict):
            errors.append(f"benchmark record is not an object: {line}")
            continue
        # Backends and launchers may emit their own structured diagnostics.
        # Only records explicitly claiming the steady-state timing protocol
        # belong to this parser.
        if record.get("kind") != "steady_state":
            continue
        actual_keys = set(record)
        if actual_keys != required_keys:
            missing = sorted(required_keys - actual_keys)
            extra = sorted(actual_keys - required_keys)
            errors.append(
                "benchmark record fields do not match schema"
                f"; missing={missing}; extra={extra}"
            )
            continue
        case = record.get("case")
        provider = record.get("provider")
        samples = record.get("samples")
        if (
            record.get("schema_version") != 1
            or record.get("kind") != "steady_state"
            or record.get("unit") != "us"
            or not isinstance(case, str)
            or not case
            or not isinstance(provider, str)
            or not provider
            or not positive_number(record.get("median"))
            or not positive_number(record.get("p90"))
            or not isinstance(samples, list)
            or not samples
            or not all(positive_number(sample) for sample in samples)
        ):
            errors.append(
                "benchmark record violates result.schema.json: " + line
            )
            continue
        ordered_samples = sorted(float(sample) for sample in samples)

        def nearest_rank(fraction: float) -> float:
            index = max(0, math.ceil(fraction * len(ordered_samples)) - 1)
            return ordered_samples[index]

        expected_median = nearest_rank(0.5)
        expected_p90 = nearest_rank(0.9)
        if not math.isclose(
            float(record["median"]),
            expected_median,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            float(record["p90"]),
            expected_p90,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            errors.append(
                f"benchmark summary does not match samples for "
                f"case={case} provider={provider}"
            )
            continue
        case_records = records.setdefault(case, {})
        if provider in case_records:
            errors.append(
                f"duplicate benchmark record for case={case} "
                f"provider={provider}"
            )
            continue
        case_records[provider] = record
    return records, errors


def convolution_case_operator(case: str) -> str | None:
    """Map a rank-qualified convolution case to its manifest operator."""
    match = HYGON_CONVOLUTION_CASE_PATTERN.match(case)
    if match is None:
        return None
    return f"conv_{match.group(1)}"


def benchmark_case_operator(
    case: str, manifest_operators: list[str]
) -> str | None:
    """Resolve a benchmark case to its longest manifest operator token."""
    matches = [
        candidate
        for candidate in dict.fromkeys(manifest_operators)
        if case == candidate or case.startswith(f"{candidate}_")
    ]
    convolution_operator = convolution_case_operator(case)
    if (
        convolution_operator is not None
        and convolution_operator in manifest_operators
    ):
        matches.append(convolution_operator)
    if not matches:
        return None
    return max(matches, key=len)


def load_hygon_comparable_case_catalog(
    manifest_operators: list[str],
    path: Path = HYGON_COMPARABLE_CASE_CATALOG,
) -> dict[str, Any]:
    """Load the repository-owned set of benchmark cases comparable to hipDNN."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot load Hygon comparable-case catalog {path}: {error}"
        ) from error

    required_keys = {
        "schema_version",
        "platform",
        "suite",
        "metric",
        "source",
        "declared_operator_count",
        "declared_case_count",
        "operators",
    }
    if not isinstance(document, dict) or set(document) != required_keys:
        actual_keys = set(document) if isinstance(document, dict) else set()
        raise RuntimeError(
            "Hygon comparable-case catalog fields do not match schema; "
            f"missing={sorted(required_keys - actual_keys)}; "
            f"extra={sorted(actual_keys - required_keys)}"
        )
    schema_version = document["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or document["platform"] != "hygon"
        or document["suite"] != "benchmark"
        or document["metric"] != HYGON_SPEEDUP_METRIC
        or not isinstance(document["source"], str)
        or not document["source"].strip()
    ):
        raise RuntimeError(
            "Hygon comparable-case catalog metadata violates schema version 1"
        )
    operators = document["operators"]
    if not isinstance(operators, dict):
        raise RuntimeError(
            "Hygon comparable-case catalog operators must be an object"
        )
    if not operators:
        raise RuntimeError(
            "Hygon comparable-case catalog must declare at least one operator"
        )

    normalized: dict[str, list[str]] = {}
    all_cases: set[str] = set()
    for operator, cases in operators.items():
        if not isinstance(operator, str) or operator not in manifest_operators:
            raise RuntimeError(
                "Hygon comparable-case catalog declares an unknown benchmark "
                f"operator: {operator!r}"
            )
        if not isinstance(cases, list) or not cases:
            raise RuntimeError(
                "Hygon comparable-case catalog operator must declare a "
                f"nonempty case list: {operator}"
            )
        normalized_cases: list[str] = []
        for case in cases:
            if not isinstance(case, str) or not case.strip() or case != case.strip():
                raise RuntimeError(
                    "Hygon comparable-case catalog contains an invalid case "
                    f"for operator={operator}: {case!r}"
                )
            owner = benchmark_case_operator(case, manifest_operators)
            if owner != operator:
                raise RuntimeError(
                    "Hygon comparable-case catalog case ownership mismatch: "
                    f"case={case} owner={owner} declared_operator={operator}"
                )
            if case in all_cases:
                raise RuntimeError(
                    "Hygon comparable-case catalog contains a duplicate case: "
                    + case
                )
            all_cases.add(case)
            normalized_cases.append(case)
        normalized[operator] = normalized_cases

    declared_operator_count = document["declared_operator_count"]
    declared_case_count = document["declared_case_count"]
    if (
        not isinstance(declared_operator_count, int)
        or isinstance(declared_operator_count, bool)
        or declared_operator_count != len(normalized)
        or not isinstance(declared_case_count, int)
        or isinstance(declared_case_count, bool)
        or declared_case_count != len(all_cases)
    ):
        raise RuntimeError(
            "Hygon comparable-case catalog declared counts do not match its "
            "operator/case contents"
        )

    return {
        **{key: document[key] for key in required_keys - {"operators"}},
        "catalog_path": str(path.resolve()),
        "operators": normalized,
    }


def hygon_comparable_coverage(
    results: dict[str, dict[str, Any]],
    selected_benchmark_operators: list[str],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Verify every selected, declared case emitted a complete provider pair."""
    declared = catalog["operators"]
    selected_declared = [
        operator
        for operator in dict.fromkeys(selected_benchmark_operators)
        if operator in declared
    ]
    missing: list[dict[str, str]] = []
    observed_required = 0
    observed_pairs = 0
    required_cases = {
        (operator, case)
        for operator in selected_declared
        for case in declared[operator]
    }
    for operator in selected_declared:
        benchmark = results.get(operator, {}).get("benchmark")
        status = (
            benchmark.get("status")
            if isinstance(benchmark, dict)
            else "not_run"
        )
        records = (
            benchmark.get("records", {})
            if isinstance(benchmark, dict)
            else {}
        )
        if not isinstance(records, dict):
            records = {}
        for case in declared[operator]:
            providers = records.get(case)
            if status == "passed" and isinstance(providers, dict) and set(
                providers
            ) == {"flagdnn", "hipdnn"}:
                observed_required += 1
                continue
            missing.append(
                {
                    "operator": operator,
                    "case": case,
                    "benchmark_status": str(status),
                }
            )

    for operator in dict.fromkeys(selected_benchmark_operators):
        benchmark = results.get(operator, {}).get("benchmark")
        if not isinstance(benchmark, dict) or benchmark.get("status") != "passed":
            continue
        records = benchmark.get("records", {})
        if not isinstance(records, dict):
            continue
        observed_pairs += sum(
            isinstance(providers, dict)
            and set(providers) == {"flagdnn", "hipdnn"}
            for providers in records.values()
        )

    return {
        "schema_version": catalog["schema_version"],
        "metric": catalog["metric"],
        "source": catalog["source"],
        "catalog_path": catalog.get(
            "catalog_path", str(HYGON_COMPARABLE_CASE_CATALOG)
        ),
        "declared_operator_count": catalog["declared_operator_count"],
        "declared_case_count": catalog["declared_case_count"],
        "selected_declared_operators": selected_declared,
        "required_case_count": len(required_cases),
        "observed_required_case_count": observed_required,
        "observed_pair_case_count": observed_pairs,
        "extra_pair_case_count": observed_pairs - observed_required,
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "verified": not missing,
    }


def validate_hygon_benchmark_pairs(
    records: dict[str, dict[str, Any]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    expected = {"flagdnn", "hipdnn"}
    if not records:
        return ["passed Hygon benchmark emitted no provider records"]
    for case, providers in records.items():
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"Hygon benchmark emitted case={case} owned_by={owner}; "
                f"expected owner={operator}"
            )
        actual = set(providers)
        if actual != expected:
            errors.append(
                f"Hygon benchmark case={case} providers={sorted(actual)}; "
                f"expected={sorted(expected)}"
            )
    return errors


def benchmark_speedup_summary(
    results: dict[str, dict[str, Any]],
    threshold: float | None,
    comparable_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize strict per-case hipDNN/FlagDNN median speedups."""
    cases: list[dict[str, Any]] = []
    for operator, operator_results in results.items():
        benchmark = operator_results.get("benchmark")
        # A failed process can emit a valid prefix of timing records before
        # aborting. Those partial records are diagnostic evidence, not a
        # complete benchmark result, and must never contribute to the gate.
        if (
            not isinstance(benchmark, dict)
            or benchmark.get("status") != "passed"
        ):
            continue
        records = benchmark.get("records", {})
        if not isinstance(records, dict):
            continue
        for case, providers in records.items():
            if not isinstance(providers, dict) or set(providers) != {
                "flagdnn",
                "hipdnn",
            }:
                continue
            flagdnn_us = float(providers["flagdnn"]["median"])
            hipdnn_us = float(providers["hipdnn"]["median"])
            cases.append(
                {
                    "operator": operator,
                    "case": case,
                    "flagdnn_median_us": flagdnn_us,
                    "hipdnn_median_us": hipdnn_us,
                    "speedup": hipdnn_us / flagdnn_us,
                }
            )
    cases.sort(key=lambda record: (record["speedup"], record["case"]))
    failures = (
        []
        if threshold is None
        else [record for record in cases if record["speedup"] < threshold]
    )
    ratio_gate_passed = threshold is None or (bool(cases) and not failures)
    coverage_gate_passed = (
        threshold is None
        if comparable_coverage is None
        else bool(comparable_coverage.get("verified"))
    )
    return {
        "metric": HYGON_SPEEDUP_METRIC,
        "threshold": threshold,
        "gate_passed": ratio_gate_passed and coverage_gate_passed,
        "ratio_gate_passed": ratio_gate_passed,
        "coverage_gate_passed": coverage_gate_passed,
        "comparable_coverage_verified": (
            None
            if comparable_coverage is None
            else bool(comparable_coverage.get("verified"))
        ),
        "case_count": len(cases),
        "passed_case_count": len(cases) - len(failures),
        "failed_case_count": len(failures),
        "minimum_speedup": cases[0]["speedup"] if cases else None,
        "failures": failures,
    }


def validate_hygon_case_accounting(
    output: str,
    operator: str,
    suite: str,
    ctest_reported_status: str,
    records: dict[str, dict[str, Any]],
    skip_records: list[dict[str, str]],
) -> list[str]:
    marker_records: list[tuple[str, str]] = []
    marker_pattern = re.compile(
        r"^(FLAGDNN_[A-Z0-9_]+_(?:FUNCTIONAL|BENCHMARK)):\s*(.*)$"
    )
    accounting_pattern = re.compile(
        r"(PASS|SKIP)\s+"
        r"(?:(?:cases)=(\d+)\s+)?executed=(\d+)\s+skipped=(\d+)\s*$"
    )
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        match = marker_pattern.fullmatch(line)
        if match is not None:
            marker_records.append((match.group(1), match.group(2)))
    marker_operator = operator.upper()
    if suite == "functional":
        marker_operator = HYGON_FUNCTIONAL_ACCOUNTING_MARKER_OVERRIDES.get(
            operator, marker_operator
        )
    expected_marker = f"FLAGDNN_{marker_operator}_{suite.upper()}"
    if len(marker_records) != 1:
        found_markers = [record[0] for record in marker_records]
        return [
            f"Hygon {suite} suite for op={operator} must emit exactly one "
            f"{expected_marker} case accounting record; found "
            f"{len(marker_records)} markers={found_markers}"
        ]
    marker, accounting = marker_records[0]
    errors: list[str] = []
    if marker != expected_marker:
        errors.append(
            f"Hygon suite accounting marker={marker}; "
            f"expected {expected_marker}"
        )
    match = accounting_pattern.fullmatch(accounting)
    if match is None:
        errors.append(
            f"Hygon suite accounting payload for marker={marker} is malformed"
        )
        return errors
    reported = match.group(1)
    matched = int(match.group(2)) if match.group(2) is not None else None
    executed = int(match.group(3))
    skipped = int(match.group(4))
    expected_reported = "SKIP" if ctest_reported_status == "skipped" else "PASS"
    if reported != expected_reported:
        errors.append(
            f"Hygon suite accounting status={reported}; "
            f"CTest status requires {expected_reported}"
        )
    if matched is None:
        errors.append(f"Hygon {suite} accounting omitted cases=<count>")
    else:
        if matched <= 0:
            errors.append("Hygon suite accounting cases must be positive")
        if matched != executed + skipped:
            errors.append(
                f"Hygon suite cases={matched} but executed+skipped="
                f"{executed + skipped}"
            )
    if reported == "PASS" and executed <= 0:
        errors.append("Hygon PASS suite must execute at least one case")
    if reported == "SKIP" and (
        executed != 0 or matched is None or skipped != matched
    ):
        errors.append(
            "Hygon SKIP suite must execute zero cases and skip every case"
        )
    if len(skip_records) != skipped:
        errors.append(
            f"Hygon suite reports skipped={skipped} but emitted "
            f"{len(skip_records)} unique structured skip records"
        )
    if suite == "benchmark":
        if len(records) != executed:
            errors.append(
                f"Hygon benchmark reports executed={executed} but emitted "
                f"{len(records)} complete case record groups"
            )
    return errors


def hipdnn_skip_records(output: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    pattern = re.compile(
        r"^\[SKIP\]\[hipdnn\]\s+op=([^\s]+)\s+"
        r"case=([^\s]+)\s+reason=(.+)$"
    )
    for raw_line in output.splitlines():
        line = re.sub(r"^\s*\d+:\s?", "", raw_line).strip()
        if not line.startswith("[SKIP][hipdnn]"):
            continue
        match = pattern.fullmatch(line)
        record = {"message": line}
        if match is not None:
            record.update(
                {
                    "op": match.group(1),
                    "case": match.group(2),
                    "reason": match.group(3),
                }
            )
        records.append(record)
    return records


def validate_hygon_skip_records(
    records: list[dict[str, str]],
    operator: str,
    manifest_operators: list[str],
) -> list[str]:
    errors: list[str] = []
    matching_records = 0
    if not records:
        return [
            "skipped Hygon suite emitted no structured "
            f"[SKIP][hipdnn] record for op={operator}"
        ]
    for index, record in enumerate(records, start=1):
        missing = [
            key
            for key in ("op", "case", "reason")
            if not record.get(key, "").strip()
        ]
        if missing:
            errors.append(
                f"hipDNN skip record {index} is missing non-empty "
                + ", ".join(missing)
            )
            continue
        if record["op"] != operator:
            errors.append(
                f"hipDNN skip record {index} has op={record['op']}; "
                f"expected op={operator}"
            )
            continue
        case = record["case"]
        owner = benchmark_case_operator(case, manifest_operators)
        if owner != operator:
            errors.append(
                f"hipDNN skip record {index} has case={case} "
                f"owned_by={owner}; expected owner={operator}"
            )
            continue
        matching_records += 1
    cases = [record.get("case", "") for record in records]
    if len(cases) != len(set(cases)):
        errors.append("hipDNN skip records contain duplicate case names")
    if matching_records == 0:
        errors.append(
            "skipped Hygon suite emitted no legal skip record matching "
            f"op={operator}"
        )
    return errors


def ctest_status(exit_code: int, output: str) -> str:
    if "No tests were found" in output:
        return "not_found"
    if exit_code != 0:
        return "failed"
    if CTEST_DISABLED_PATTERN.search(output) is not None:
        return "failed"
    if CTEST_SKIPPED_PATTERN.search(output) is not None:
        return "skipped"
    return "passed"


def verbose_ctest_command(
    build_dir: Path,
    expression: str,
    configuration: str | None = None,
) -> list[str]:
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "-j1",
        "-R",
        expression,
        "-V",
    ]
    if configuration is not None:
        command.extend(["-C", configuration])
    return command


def ctest_command(
    build_dir: Path,
    operator: str,
    suite: str,
    platform: str,
    configuration: str | None = None,
) -> list[str]:
    test_name = f"{suite}.{platform}.{operator}"
    expression = f"^{re.escape(test_name)}$"
    if platform == "nvidia" and suite == "functional" and operator == "matmul":
        expression = r"^functional\.nvidia\.matmul(\.host)?$"
    return verbose_ctest_command(build_dir, expression, configuration)


class _TerminationSignal(SystemExit):
    def __init__(self, signum: int):
        super().__init__(128 + signum)
        self.signum = signum


def _linux_process_identity(pid: int) -> tuple[str, int, int] | None:
    """Return (state, process-group, session) from procfs, if available."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError):
        return None
    # The comm field is parenthesized and may itself contain whitespace or
    # parentheses.  Splitting after its final ')' keeps the fixed stat fields
    # aligned: state, ppid, pgrp, session, ...
    comm_end = stat.rfind(")")
    if comm_end < 0:
        return None
    fields = stat[comm_end + 1 :].split()
    if len(fields) < 4:
        return None
    try:
        return fields[0], int(fields[2]), int(fields[3])
    except ValueError:
        return None


def _session_process_groups(session_id: int) -> dict[int, set[int]]:
    """Snapshot live process groups belonging to one Linux/POSIX session."""
    if session_id <= 0:
        return {}
    try:
        if session_id == os.getsid(0):
            # Never allow cleanup of the runner's own session.
            return {}
    except OSError:
        return {}

    groups: dict[int, set[int]] = {}
    proc = Path("/proc")
    if proc.is_dir():
        try:
            entries = tuple(proc.iterdir())
        except OSError:
            entries = ()
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            identity = _linux_process_identity(pid)
            if identity is None:
                continue
            state, process_group, process_session = identity
            if (
                process_session == session_id
                and process_group > 0
                and state not in {"Z", "X"}
            ):
                groups.setdefault(process_group, set()).add(pid)
        return groups

    # procfs is the race-resistant implementation used on Linux.  Retain a
    # conservative POSIX fallback for other hosts: start_new_session=True
    # guarantees that the direct child initially has PGID == SID == PID.
    try:
        if (
            os.getsid(session_id) == session_id
            and os.getpgid(session_id) == session_id
        ):
            groups[session_id] = {session_id}
    except (OSError, ProcessLookupError):
        pass
    return groups


def _signal_session_process_groups(
    session_id: int,
    signum: signal.Signals,
    previously_signaled: dict[int, set[int]],
) -> None:
    """Signal each newly observed group in a private child session."""
    groups = _session_process_groups(session_id)
    own_process_group = os.getpgrp()
    # Descendant groups are signaled first.  For SIGKILL this gives their
    # supervisor a brief opportunity to reap them before the session leader
    # is killed as well.
    ordered_groups = sorted(groups, key=lambda group: group == session_id)
    killed_descendant_group = False
    for process_group in ordered_groups:
        members = groups[process_group]
        if process_group == own_process_group:
            continue
        if members.issubset(previously_signaled.get(process_group, set())):
            continue
        # Re-snapshot immediately before killpg.  This narrows the unavoidable
        # PID/PGID reuse race and prevents a stale group id from escaping the
        # target session.
        current_members = _session_process_groups(session_id).get(
            process_group, set()
        )
        if not current_members:
            continue
        if (
            signum == signal.SIGKILL
            and process_group == session_id
            and killed_descendant_group
        ):
            time.sleep(PROCESS_CLEANUP_POLL_SECONDS)
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            continue
        previously_signaled.setdefault(process_group, set()).update(
            current_members
        )
        if signum == signal.SIGKILL and process_group != session_id:
            killed_descendant_group = True


def _terminate_process_session(
    process: subprocess.Popen[str], session_id: int
) -> tuple[str, str]:
    """Terminate and reap every process group in a spawned test session."""
    latest_output = ""
    latest_error = ""

    def cleanup_phase(signum: signal.Signals, grace_seconds: float) -> bool:
        nonlocal latest_output, latest_error
        deadline = time.monotonic() + grace_seconds
        signaled: dict[int, set[int]] = {}
        while True:
            _signal_session_process_groups(session_id, signum, signaled)
            remaining = deadline - time.monotonic()
            wait_seconds = max(
                0.001,
                min(PROCESS_CLEANUP_POLL_SECONDS, max(0.0, remaining)),
            )
            try:
                latest_output, latest_error = process.communicate(
                    timeout=wait_seconds
                )
            except subprocess.TimeoutExpired:
                pass
            if (
                process.poll() is not None
                and not _session_process_groups(session_id)
            ):
                if not latest_output and not latest_error:
                    latest_output, latest_error = process.communicate()
                return True
            if remaining <= 0.0:
                return False

    if cleanup_phase(signal.SIGTERM, PROCESS_TERMINATION_GRACE_SECONDS):
        return latest_output, latest_error
    if cleanup_phase(signal.SIGKILL, PROCESS_KILL_GRACE_SECONDS):
        return latest_output, latest_error

    # An uninterruptible descendant can keep an inherited pipe open.  Do not
    # hang the runner forever after both bounded cleanup phases.  The direct
    # child is still forcibly reaped, while captured prefix output remains
    # available for diagnostics.
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=PROCESS_CLEANUP_POLL_SECONDS)
    except subprocess.TimeoutExpired as error:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        try:
            process.wait(timeout=PROCESS_CLEANUP_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        output = error.output if isinstance(error.output, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return output or latest_output, stderr or latest_error


def run_process_group(
    command: list[str],
    environment: dict[str, str],
    timeout: int,
) -> tuple[str, str, int | None, bool]:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    previous_handlers: dict[signal.Signals, Any] = {}

    def handle_termination(signum: int, _frame: Any) -> None:
        raise _TerminationSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_termination)

    session_id = process.pid

    def terminate() -> tuple[str, str]:
        return _terminate_process_session(process, session_id)

    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout, stderr, process.returncode, False
        except subprocess.TimeoutExpired:
            stdout, stderr = terminate()
            return stdout, stderr, None, True
        except BaseException:
            try:
                terminate()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise

    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_one(
    build_dir: Path,
    operator: str,
    suite: str,
    platform: str,
    environment: dict[str, str],
    timeout: int,
    verbose: bool,
    manifest_operators: list[str],
    configuration: str | None = None,
) -> dict[str, Any]:
    command = ctest_command(
        build_dir, operator, suite, platform, configuration
    )
    started = time.monotonic()
    stdout, stderr, exit_code, timed_out = run_process_group(
        command, environment, timeout
    )
    if timed_out:
        status = "timeout"
    else:
        status = ctest_status(exit_code, stdout + "\n" + stderr)
    ctest_reported_status = status

    duration = time.monotonic() - started
    result: dict[str, Any] = {
        "status": status,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "command": command,
    }
    combined_output = stdout + "\n" + stderr
    skip_records = hipdnn_skip_records(combined_output)
    if skip_records:
        result["skip_records"] = skip_records
    if platform == "hygon" and (
        ctest_reported_status == "skipped" or skip_records
    ):
        skip_record_errors = validate_hygon_skip_records(
            skip_records, operator, manifest_operators
        )
        if skip_record_errors:
            result["skip_record_errors"] = skip_record_errors
            status = "failed"
            result["status"] = status
    records: dict[str, dict[str, Any]] = {}
    if suite == "benchmark":
        records, record_errors = benchmark_records(combined_output)
        if ctest_reported_status == "passed" and not records:
            record_errors.append(
                "passed benchmark emitted no steady-state timing records"
            )
        if status == "passed" and platform == "hygon":
            record_errors.extend(
                validate_hygon_benchmark_pairs(
                    records, operator, manifest_operators
                )
            )
        if ctest_reported_status == "skipped" and records:
            record_errors.append(
                "skipped benchmark emitted timing provider records"
            )
        result["records"] = records
        if record_errors:
            result["record_errors"] = record_errors
            if status in ("passed", "skipped"):
                status = "failed"
                result["status"] = status
    if (
        platform == "hygon"
        and ctest_reported_status in ("passed", "skipped")
    ):
        accounting_errors = validate_hygon_case_accounting(
            combined_output,
            operator,
            suite,
            ctest_reported_status,
            records,
            skip_records,
        )
        if accounting_errors:
            result["case_accounting_errors"] = accounting_errors
            status = "failed"
            result["status"] = status
    if verbose or status != "passed":
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(
                stderr,
                file=sys.stderr,
                end="" if stderr.endswith("\n") else "\n",
            )
        for error in result.get("record_errors", []):
            print(f"benchmark record error: {error}", file=sys.stderr)
        for error in result.get("skip_record_errors", []):
            print(f"hipDNN skip record error: {error}", file=sys.stderr)
        for error in result.get("case_accounting_errors", []):
            print(f"Hygon case accounting error: {error}", file=sys.stderr)
    return result


def required_preflight_tests(
    platform: str, suites: list[str] | tuple[str, ...] = VALID_SUITES
) -> set[str]:
    required_tests = {
        "core.json_contract",
        "core.backend_dependency_boundary",
        "core.backend_contract",
        "core.c_header",
        "core.c_api",
        "core.convolution_backward_api",
        "core.batchnorm_inference_api",
        "core.cpp_header",
        "core.frontend_api",
        "core.test_architecture",
        "core.run_tests_contract",
        "core.default_backend_contract",
    }
    if "benchmark" in suites:
        required_tests.update(
            {
                "benchmark.catalog_contract",
                "benchmark.catalog_dependency_boundary",
            }
        )
    if platform == "hygon":
        required_tests.update(
            {
                "integration.hygon.validation_contract",
                "integration.hygon.convolution_validation_static_contract",
                "integration.hygon.cmake_configuration_contract",
                "integration.hygon.dependency_boundary",
                "integration.hygon.compiler_contract",
                "integration.hygon.reference_dependency_boundary",
                "integration.hygon.jit",
                "integration.hygon.pointwise_smoke",
                "integration.hygon.installed_consumer",
                "integration.hygon.runtime",
                "integration.hygon.graph",
                "integration.hygon.jit_candidate_compatibility_contract",
                "integration.hygon.jit_global_state_contract",
                "integration.hygon.normalization_stability",
            }
        )
    elif platform == "nvidia":
        required_tests.update(
            {
                "integration.nvidia.dependency_boundary",
                "integration.nvidia.reference_dependency_boundary",
                "integration.nvidia.runtime",
                "integration.nvidia.graph",
            }
        )
    return required_tests


def run_preflight(
    build_dir: Path,
    platform: str,
    environment: dict[str, str],
    timeout: int,
    verbose: bool,
    suites: list[str],
    configuration: str | None = None,
) -> dict[str, Any]:
    visibility_masks = {
        variable: environment[variable]
        for variable in HYGON_VISIBILITY_VARIABLES
        if variable in environment
    }
    required_tests = required_preflight_tests(platform, suites)

    listing_command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--show-only=json-v1",
    ]
    if configuration is not None:
        listing_command.extend(["-C", configuration])
    listing_stdout, listing_stderr, listing_exit, listing_timeout = (
        run_process_group(listing_command, environment, min(timeout, 60))
    )
    listing_error: str | None = None
    discovered_tests: set[str] = set()
    if listing_timeout:
        listing_error = "CTest catalog listing timed out"
    elif listing_exit != 0:
        listing_error = "CTest catalog listing failed: " + (
            listing_stderr.strip() or listing_stdout.strip()
        )
    else:
        try:
            catalog = json.loads(listing_stdout)
            discovered_tests = {
                test["name"]
                for test in catalog.get("tests", [])
                if isinstance(test, dict) and isinstance(test.get("name"), str)
            }
        except (AttributeError, json.JSONDecodeError, TypeError) as error:
            listing_error = f"cannot parse CTest JSON catalog: {error}"
    missing_tests = sorted(required_tests - discovered_tests)
    if listing_error is not None or missing_tests:
        errors = []
        if listing_error is not None:
            errors.append(listing_error)
        if missing_tests:
            errors.append(
                "required preflight tests are missing: "
                + ", ".join(missing_tests)
            )
        for error in errors:
            print(f"preflight error: {error}", file=sys.stderr)
        return {
            "status": "failed",
            "duration_seconds": 0.0,
            "exit_code": listing_exit,
            "command": listing_command,
            "required_tests": sorted(required_tests),
            "missing_tests": missing_tests,
            "errors": errors,
            "visibility_masks": visibility_masks,
        }

    patterns = [r"core\."]
    if "benchmark" in suites:
        patterns.append(r"benchmark\.catalog_")
    patterns.append(rf"integration\.{re.escape(platform)}\.")
    expression = "^(" + "|".join(patterns) + ")"
    command = verbose_ctest_command(build_dir, expression, configuration)
    started = time.monotonic()
    stdout, stderr, exit_code, timed_out = run_process_group(
        command, environment, timeout
    )
    status = (
        "timeout"
        if timed_out
        else ctest_status(exit_code, stdout + "\n" + stderr)
    )
    duration = time.monotonic() - started
    if verbose or status != "passed":
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(
                stderr,
                file=sys.stderr,
                end="" if stderr.endswith("\n") else "\n",
            )
    return {
        "status": status,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "command": command,
        "required_tests": sorted(required_tests),
        "missing_tests": [],
        "errors": [],
        "visibility_masks": visibility_masks,
    }


def summary_status_fields(
    overall_status: str,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "overall_status": overall_status,
        "exit_code": exit_code,
    }


def summary_envelope(
    arguments: argparse.Namespace,
    overall_status: str,
    exit_code: int | None,
) -> dict[str, Any]:
    """Build the stable top-level fields shared by every summary state."""
    return {
        **summary_status_fields(overall_status, exit_code),
        "platform": arguments.platform,
        "device": arguments.device,
    }


def preliminary_output_argument(arguments: list[str]) -> Path | None:
    """Find the last explicit --output before full argparse validation."""
    requested: Path | None = None
    for index, argument in enumerate(arguments):
        if argument.startswith("--output="):
            requested = Path(argument.partition("=")[2])
        elif (
            argument == "--output"
            and index + 1 < len(arguments)
            and not arguments[index + 1].startswith("-")
        ):
            requested = Path(arguments[index + 1])
    return requested


def atomic_write_summary(output: Path, summary: dict[str, Any]) -> None:
    """Publish one complete JSON document without exposing partial writes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Run FlagDNN native functional tests and benchmarks " "serially"
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help=(
            "configured CMake build directory "
            "(default: FLAGDNN_BUILD_DIR or build/<platform>)"
        ),
    )
    parser.add_argument(
        "--ops",
        help=(
            "comma-separated operators, or 'all' for every manifest operator "
            f"(default: {','.join(DEFAULT_OPERATORS)})"
        ),
    )
    parser.add_argument("--op-list-file", type=Path)
    parser.add_argument(
        "--suites",
        default="functional",
        help="functional, benchmark, comma-separated values, or all",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        help=(
            "Hygon benchmark-only per-case minimum median speedup, defined "
            "as hipDNN_us/FlagDNN_us"
        ),
    )
    parser.add_argument(
        "--platform",
        default=os.environ.get("FLAGDNN_BENCHMARK_PLATFORM", "nvidia"),
        help="benchmark CTest platform component",
    )
    parser.add_argument(
        "--device",
        help=(
            "visible device id; leaves the current environment unchanged "
            "if omitted"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="timeout in seconds for each operator/suite",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("FLAGDNN_BUILD_TYPE"),
        help=(
            "CMake build configuration for CTest (default: the configuration "
            "recorded by tools/build.sh, the single-config CMAKE_BUILD_TYPE, "
            "or Release for a multi-config tree)"
        ),
    )
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument(
        "--preflight",
        dest="preflight",
        action="store_true",
        help="run core and platform integration contracts before operators",
    )
    preflight.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="skip core and platform integration contracts",
    )
    parser.set_defaults(preflight=None)
    parser.add_argument(
        "--output", type=Path, help="optional JSON summary path"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list every manifest operator and exit",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _run_main() -> int:
    preliminary_output = preliminary_output_argument(sys.argv[1:])
    output: Path | None = None
    if preliminary_output is not None:
        try:
            output = preliminary_output.expanduser().resolve()
            atomic_write_summary(
                output,
                {
                    **summary_status_fields("running", None),
                    "phase": "argument_validation",
                },
            )
        except (OSError, RuntimeError) as error:
            print(
                f"error: cannot initialize --output summary: {error}",
                file=sys.stderr,
            )
            return 2
    try:
        arguments = parse_arguments()
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 2
        if output is not None:
            try:
                atomic_write_summary(
                    output,
                    {
                        **summary_status_fields(
                            "passed" if exit_code == 0 else "failed",
                            exit_code,
                        ),
                        "phase": "argument_validation",
                        **({"mode": "help"} if exit_code == 0 else {}),
                    },
                )
            except OSError as summary_error:
                print(
                    "error: cannot finalize --output summary: "
                    f"{summary_error}",
                    file=sys.stderr,
                )
                return 2
        return exit_code
    if arguments.output is not None:
        parsed_output = arguments.output.expanduser().resolve()
        if output is None or parsed_output != output:
            print(
                "error: could not safely identify --output before argument "
                "validation",
                file=sys.stderr,
            )
            return 2
        atomic_write_summary(
            output,
            summary_envelope(arguments, "running", None),
        )

    def finish_early(
        exit_code: int,
        overall_status: str,
        **fields: Any,
    ) -> int:
        if output is not None:
            summary = summary_envelope(
                arguments, overall_status, exit_code
            )
            summary.update(fields)
            try:
                atomic_write_summary(output, summary)
            except OSError as error:
                print(
                    f"error: cannot finalize --output summary: {error}",
                    file=sys.stderr,
                )
                return 2
        return exit_code

    def validation_error(message: str) -> int:
        print(f"error: {message}", file=sys.stderr)
        return finish_early(2, "failed", error=message)

    if PLATFORM_PATTERN.fullmatch(arguments.platform) is None:
        return validation_error(
            "--platform must match [a-z][a-z0-9_]*"
        )
    try:
        manifests = operator_manifests()
        suites = requested_suites(arguments.suites)
        if arguments.list:
            listed = dict.fromkeys(
                operator for suite in suites for operator in manifests[suite]
            )
            print("\n".join(listed))
            return finish_early(
                0,
                "passed",
                mode="list",
                suites=suites,
                operators=list(listed),
            )
        suite_operators = requested_operators(
            manifests,
            suites,
            arguments.ops,
            arguments.op_list_file,
        )
    except (OSError, RuntimeError, ValueError) as error:
        return validation_error(str(error))

    build_dir = resolve_build_directory(
        arguments.build_dir, arguments.platform
    )
    try:
        validate_build_directory(build_dir)
        build_configuration = resolve_build_configuration(
            build_dir, arguments.config
        )
    except (OSError, ValueError) as error:
        return validation_error(str(error))
    if arguments.timeout <= 0:
        return validation_error("--timeout must be positive")
    if arguments.min_speedup is not None and (
        not math.isfinite(arguments.min_speedup)
        or arguments.min_speedup <= 0.0
    ):
        return validation_error("--min-speedup must be finite and positive")
    if arguments.min_speedup is not None and arguments.platform != "hygon":
        return validation_error("--min-speedup is Hygon-only")
    if arguments.min_speedup is not None and "benchmark" not in suites:
        return validation_error("--min-speedup requires the benchmark suite")
    environment = device_environment(arguments.platform, arguments.device)
    manifest_operators = list(
        dict.fromkeys(
            operator
            for manifest in manifests.values()
            for operator in manifest
        )
    )
    hygon_comparable_catalog: dict[str, Any] | None = None
    if arguments.platform == "hygon" and "benchmark" in suites:
        try:
            hygon_comparable_catalog = load_hygon_comparable_case_catalog(
                manifests["benchmark"]
            )
        except RuntimeError as error:
            return validation_error(str(error))
    operators = list(
        dict.fromkeys(
            operator for suite in suites for operator in suite_operators[suite]
        )
    )
    total = sum(len(suite_operators[suite]) for suite in suites)
    if total == 0:
        return validation_error(
            "operator selection produced zero operator/suite runs"
        )
    results: dict[str, dict[str, Any]] = {}
    failed = False
    preflight_result: dict[str, Any] | None = None
    should_run_preflight = (
        arguments.preflight
        if arguments.preflight is not None
        else arguments.platform == "hygon"
    )
    if should_run_preflight:
        print(
            "[preflight] core and platform integration contracts", flush=True
        )
        preflight_result = run_preflight(
            build_dir=build_dir,
            platform=arguments.platform,
            environment=environment,
            timeout=arguments.timeout,
            verbose=arguments.verbose,
            suites=suites,
            configuration=build_configuration,
        )
        preflight_status = preflight_result["status"]
        print(
            "  "
            f"{preflight_status}: "
            f"{preflight_result['duration_seconds']:.2f}s",
            flush=True,
        )
        failed = preflight_status != "passed"
    completed_count = 0

    if not failed:
        for suite in suites:
            for operator in suite_operators[suite]:
                completed_count += 1
                print(
                    f"[{completed_count}/{total}] {suite} {operator}",
                    flush=True,
                )
                result = run_one(
                    build_dir=build_dir,
                    operator=operator,
                    suite=suite,
                    platform=arguments.platform,
                    environment=environment,
                    timeout=arguments.timeout,
                    verbose=arguments.verbose,
                    manifest_operators=manifest_operators,
                    configuration=build_configuration,
                )
                results.setdefault(operator, {})[suite] = result
                status = result["status"]
                duration = result["duration_seconds"]
                print(f"  {status}: {duration:.2f}s", flush=True)
                expected_skip = (
                    arguments.platform == "hygon" and status == "skipped"
                )
                failed = failed or (
                    status != "passed" and not expected_skip
                )

    comparable_coverage: dict[str, Any] | None = None
    if hygon_comparable_catalog is not None and not (
        preflight_result is not None
        and preflight_result["status"] != "passed"
    ):
        comparable_coverage = hygon_comparable_coverage(
            results,
            suite_operators["benchmark"],
            hygon_comparable_catalog,
        )
        if not comparable_coverage["verified"]:
            print(
                "comparable coverage gate: "
                f"{comparable_coverage['observed_required_case_count']}/"
                f"{comparable_coverage['required_case_count']} declared cases "
                "emitted complete FlagDNN/hipDNN pairs",
                flush=True,
            )
            for missing in comparable_coverage["missing_cases"]:
                print(
                    "  FAIL missing comparable pair "
                    f"op={missing['operator']} case={missing['case']} "
                    f"benchmark_status={missing['benchmark_status']}",
                    flush=True,
                )
            failed = True

    status_counts = {
        status: sum(
            result["status"] == status
            for operator_results in results.values()
            for result in operator_results.values()
        )
        for status in ("passed", "failed", "skipped", "timeout", "not_found")
    }
    benchmark_case_pairs = 0
    benchmark_provider_records = 0
    benchmark_record_errors = 0
    hipdnn_skip_record_errors = 0
    hygon_case_accounting_errors = 0
    hipdnn_reference_skips = 0
    for operator_results in results.values():
        for result in operator_results.values():
            if result["status"] == "passed":
                records = result.get("records", {})
                benchmark_provider_records += sum(
                    len(providers) for providers in records.values()
                )
                benchmark_case_pairs += sum(
                    set(providers) == {"flagdnn", "hipdnn"}
                    for providers in records.values()
                )
            benchmark_record_errors += len(result.get("record_errors", []))
            hipdnn_skip_record_errors += len(
                result.get("skip_record_errors", [])
            )
            hygon_case_accounting_errors += len(
                result.get("case_accounting_errors", [])
            )
            if result["status"] in ("passed", "skipped"):
                hipdnn_reference_skips += len(result.get("skip_records", []))

    performance = (
        benchmark_speedup_summary(
            results,
            arguments.min_speedup,
            comparable_coverage,
        )
        if arguments.platform == "hygon" and "benchmark" in suites
        else None
    )
    if arguments.min_speedup is not None:
        assert performance is not None
        failures = performance["failures"]
        print(
            "performance gate: "
            f"{performance['passed_case_count']}/"
            f"{performance['case_count']} cases meet "
            f"speedup >= {arguments.min_speedup:.6g}",
            flush=True,
        )
        for record in failures:
            print(
                "  FAIL "
                f"op={record['operator']} case={record['case']} "
                f"flagdnn_us={record['flagdnn_median_us']:.9g} "
                f"hipdnn_us={record['hipdnn_median_us']:.9g} "
                f"speedup={record['speedup']:.9g}",
                flush=True,
            )
        if performance["case_count"] == 0:
            print(
                "  FAIL no comparable FlagDNN/hipDNN benchmark case was "
                "emitted",
                flush=True,
            )
        failed = failed or not performance["gate_passed"]

    final_exit_code = 1 if failed else 0
    summary = {
        **summary_envelope(
            arguments,
            "failed" if failed else "passed",
            final_exit_code,
        ),
        "build_dir": str(build_dir),
        "build_configuration": build_configuration,
        "operators": operators,
        "suites": suites,
        "suite_operators": suite_operators,
        "preflight": preflight_result,
        "status_counts": status_counts,
        "coverage": {
            "benchmark_case_pairs": benchmark_case_pairs,
            "benchmark_provider_records": benchmark_provider_records,
            "benchmark_record_errors": benchmark_record_errors,
            "hipdnn_reference_skips": hipdnn_reference_skips,
            "hipdnn_skip_record_errors": hipdnn_skip_record_errors,
            "hygon_case_accounting_errors": hygon_case_accounting_errors,
        },
        "comparable_coverage": comparable_coverage,
        "performance": performance,
        "results": results,
    }
    if output is not None:
        try:
            atomic_write_summary(output, summary)
        except OSError as error:
            print(
                f"error: cannot finalize --output summary: {error}",
                file=sys.stderr,
            )
            return 2
        print(f"summary: {output}")

    return final_exit_code


def main() -> int:
    previous_handlers: dict[signal.Signals, Any] = {}

    def handle_termination(signum: int, _frame: Any) -> None:
        raise _TerminationSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_termination)
    try:
        return _run_main()
    except _TerminationSignal as termination:
        exit_code = 128 + termination.signum
        requested_output = preliminary_output_argument(sys.argv[1:])
        if requested_output is not None:
            try:
                atomic_write_summary(
                    requested_output.expanduser().resolve(),
                    {
                        **summary_status_fields("failed", exit_code),
                        "phase": "signal",
                        "signal": signal.Signals(termination.signum).name,
                    },
                )
            except (OSError, RuntimeError) as error:
                print(
                    f"error: cannot finalize --output summary: {error}",
                    file=sys.stderr,
                )
        return exit_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
