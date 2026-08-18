"""NVIDIA policy hooks for the repository batch-test runner."""

from __future__ import annotations


DEFAULT_TIMEOUT = 1800
PREFLIGHT_BY_DEFAULT = False
SUPPORTS_MIN_SPEEDUP = False
FILTER_REGISTERED_TESTS = False


def configure_environment(
    environment: dict[str, str], device: str | None
) -> None:
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device


def test_expression(suite: str, operator: str) -> str | None:
    if suite == "functional" and operator == "matmul":
        return r"^functional\.nvidia\.matmul(\.host)?$"
    return None


def preflight_tests(_suites: list[str] | tuple[str, ...]) -> set[str]:
    return {
        "integration.nvidia.dependency_boundary",
        "integration.nvidia.reference_dependency_boundary",
        "integration.nvidia.runtime",
        "integration.nvidia.graph",
    }
