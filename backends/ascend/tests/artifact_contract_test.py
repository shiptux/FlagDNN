#!/usr/bin/env python3
# mypy: disable-error-code=import-not-found
"""No-device Graph-to-Ascend-artifact semantic contract harness."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / "compiler"))

from flagdnn_codegen.provider_loader import get_provider  # noqa: E402
from backends.ascend.compiler_identity import (  # noqa: E402
    SUPPORTED_CODEGEN_ARCHES,
    aicore_count_from_target,
)


POINTWISE_MODES = {
    "add": 1,
    "sub": 17,
    "mul": 18,
    "div": 19,
    "min": 20,
    "max": 21,
    "mod": 22,
    "pow": 23,
    "sigmoid_backward": 40,
}
COMPARISON_POINTWISE_MODES = {
    "cmp_eq": 25,
    "cmp_neq": 26,
    "cmp_gt": 27,
    "cmp_ge": 28,
    "cmp_lt": 29,
    "cmp_le": 30,
}
LOGICAL_BINARY_POINTWISE_MODES = {"logical_and": 31, "logical_or": 32}
UNARY_POINTWISE_MODES = {
    "relu": 2,
    "sqrt": 3,
    "erf": 4,
    "identity": 5,
    "exp": 6,
    "log": 7,
    "neg": 8,
    "abs": 9,
    "ceil": 10,
    "cos": 11,
    "floor": 12,
    "rsqrt": 13,
    "sin": 14,
    "tan": 15,
    "reciprocal": 16,
    "sigmoid": 33,
    "tanh": 34,
    "elu": 35,
    "gelu": 36,
    "softplus": 37,
    "swish": 38,
    "gelu_approx_tanh": 39,
}
LOGICAL_UNARY_POINTWISE_MODES = {"logical_not": 24}
TERNARY_POINTWISE_MODES = {"binary_select": 41}
LAYOUT_OPERATIONS = ("reshape", "transpose", "slice")
REDUCTION_MODES = {"reduction_sum": 0, "reduction_avg": 1, "reduction_mul": 2}


def _tensor(
    uid: int,
    dimensions: list[int],
    strides: list[int],
    *,
    data_type: str = "float32",
    virtual: bool = False,
) -> dict[str, object]:
    return {
        "uid": uid,
        "data_type": data_type,
        "dimensions": dimensions,
        "strides": strides,
        "alignment": 16,
        "virtual": virtual,
    }


def _contiguous_strides(dimensions: list[int]) -> list[int]:
    stride = 1
    result = [0] * len(dimensions)
    for axis in range(len(dimensions) - 1, -1, -1):
        result[axis] = stride
        stride *= dimensions[axis]
    return result


def _request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float32",
    compute_data_type: str = "float32",
    autotune: bool = False,
    operations: tuple[str, str] = ("add", "add"),
    alphas: tuple[float, float] = (1.0, 1.0),
) -> dict[str, object]:
    if any(operation not in POINTWISE_MODES for operation in operations):
        raise ValueError("unsupported artifact contract fixture operation")
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_artifact_graph_contract",
            "tensor_count": 5,
            "tensors": [
                _tensor(1, [2, 1], [1, 0], data_type=storage_data_type),
                _tensor(2, [1, 3], [0, 1], data_type=storage_data_type),
                _tensor(
                    3,
                    [2, 3],
                    [4, 1],
                    data_type=storage_data_type,
                    virtual=True,
                ),
                _tensor(4, [2, 3], [3, 1], data_type=storage_data_type),
                _tensor(5, [2, 3], [3, 1], data_type=storage_data_type),
            ],
            "node_count": 2,
            "nodes": [
                {
                    "id": 0,
                    "type": operations[0],
                    "compute_data_type": compute_data_type,
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": POINTWISE_MODES[operations[0]],
                        "alpha": alphas[0],
                    },
                },
                {
                    "id": 1,
                    "type": operations[1],
                    "compute_data_type": compute_data_type,
                    "inputs": [
                        {"name": "left", "uid": 3},
                        {"name": "right", "uid": 4},
                    ],
                    "outputs": [{"name": "output", "uid": 5}],
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": POINTWISE_MODES[operations[1]],
                        "alpha": alphas[1],
                    },
                },
            ],
        },
    }


def _canonical_request(document: dict[str, object]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _mixed_binary_request(
    identity: str, target: str, *, autotune: bool = False
) -> dict[str, object]:
    return _request(
        identity,
        target,
        autotune=autotune,
        operations=("sub", "mul"),
        alphas=(2.0, 1.0),
    )


def _sigmoid_backward_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float32",
    autotune: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_sigmoid_backward_artifact_contract",
            "tensor_count": 3,
            "tensors": [
                _tensor(1, [2, 3], [4, 1], data_type=storage_data_type),
                _tensor(2, [2, 3], [5, 1], data_type=storage_data_type),
                _tensor(3, [2, 3], [6, 1], data_type=storage_data_type),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "sigmoid_backward",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": 40,
                        "alpha": 1.0,
                    },
                }
            ],
        },
    }


def _mixed_sigmoid_backward_request(
    identity: str, target: str
) -> dict[str, object]:
    result = _sigmoid_backward_request(identity, target, autotune=True)
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_mixed_sigmoid_backward_add_artifact_contract"
    graph["tensor_count"] = 5
    graph["tensors"] = [
        _tensor(1, [2, 3], [4, 1]),
        _tensor(2, [2, 3], [5, 1]),
        _tensor(3, [2, 3], [6, 1], virtual=True),
        _tensor(4, [2, 3], [3, 1]),
        _tensor(5, [2, 3], [3, 1]),
    ]
    first_node = graph["nodes"][0]
    assert isinstance(first_node, dict)
    first_node["outputs"] = [{"name": "output", "uid": 3}]
    graph["node_count"] = 2
    graph["nodes"].append(
        {
            "id": 1,
            "type": "add",
            "compute_data_type": "float32",
            "inputs": [
                {"name": "left", "uid": 3},
                {"name": "right", "uid": 4},
            ],
            "outputs": [{"name": "output", "uid": 5}],
            "attributes": {
                "n_elements": 6,
                "pointwise_mode": 1,
                "alpha": 1.0,
            },
        }
    )
    return result


def _logical_not_request(
    identity: str, target: str, *, autotune: bool = True
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_logical_not_artifact_contract",
            "tensor_count": 2,
            "tensors": [
                _tensor(1, [2, 3], [4, 1], data_type="boolean"),
                _tensor(2, [2, 3], [5, 1], data_type="boolean"),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "logical_not",
                    "compute_data_type": "boolean",
                    "inputs": [{"name": "input", "uid": 1}],
                    "outputs": [{"name": "output", "uid": 2}],
                    "attributes": {
                        "n_elements": 6,
                        "mode": 24,
                        "pointwise_mode": 24,
                    },
                }
            ],
        },
    }


def _logical_binary_request(
    identity: str,
    target: str,
    *,
    operation: str,
    autotune: bool = True,
) -> dict[str, object]:
    mode = LOGICAL_BINARY_POINTWISE_MODES[operation]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"ascend_{operation}_artifact_contract",
            "tensor_count": 3,
            "tensors": [
                _tensor(1, [2, 1], [2, 0], data_type="boolean"),
                _tensor(2, [1, 3], [0, 1], data_type="boolean"),
                _tensor(3, [2, 3], [4, 1], data_type="boolean"),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "compute_data_type": "boolean",
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": mode,
                        "alpha": 1.0,
                    },
                }
            ],
        },
    }


def _mixed_logical_request(identity: str, target: str) -> dict[str, object]:
    result = _logical_binary_request(
        identity, target, operation="logical_and", autotune=True
    )
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_mixed_logical_bool_artifact_contract"
    graph["tensor_count"] = 6
    graph["tensors"] = [
        _tensor(1, [2, 1], [1, 0], data_type="boolean"),
        _tensor(2, [1, 3], [0, 1], data_type="boolean"),
        _tensor(3, [2, 3], [3, 1], data_type="boolean", virtual=True),
        _tensor(4, [2, 3], [3, 1], data_type="boolean", virtual=True),
        _tensor(5, [2, 3], [3, 1], data_type="boolean"),
        _tensor(6, [2, 3], [4, 1], data_type="boolean"),
    ]
    graph["node_count"] = 3
    graph["nodes"] = [
        {
            "id": 0,
            "type": "logical_and",
            "compute_data_type": "boolean",
            "inputs": [
                {"name": "left", "uid": 1},
                {"name": "right", "uid": 2},
            ],
            "outputs": [{"name": "output", "uid": 3}],
            "attributes": {
                "n_elements": 6,
                "pointwise_mode": 31,
                "alpha": 1.0,
            },
        },
        {
            "id": 1,
            "type": "logical_not",
            "compute_data_type": "boolean",
            "inputs": [{"name": "input", "uid": 3}],
            "outputs": [{"name": "output", "uid": 4}],
            "attributes": {
                "n_elements": 6,
                "mode": 24,
                "pointwise_mode": 24,
            },
        },
        {
            "id": 2,
            "type": "logical_or",
            "compute_data_type": "boolean",
            "inputs": [
                {"name": "left", "uid": 4},
                {"name": "right", "uid": 5},
            ],
            "outputs": [{"name": "output", "uid": 6}],
            "attributes": {
                "n_elements": 6,
                "pointwise_mode": 32,
                "alpha": 1.0,
            },
        },
    ]
    return result


def _comparison_request(
    identity: str,
    target: str,
    *,
    operation: str = "cmp_eq",
    input_data_type: str = "float32",
    output_data_type: str = "boolean",
    compute_data_type: str = "boolean",
    autotune: bool = True,
) -> dict[str, object]:
    mode = COMPARISON_POINTWISE_MODES[operation]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"ascend_{operation}_artifact_contract",
            "tensor_count": 3,
            "tensors": [
                _tensor(1, [2, 1], [2, 0], data_type=input_data_type),
                _tensor(2, [1, 3], [0, 1], data_type=input_data_type),
                _tensor(3, [2, 3], [4, 1], data_type=output_data_type),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "compute_data_type": compute_data_type,
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": mode,
                        "alpha": 1.0,
                    },
                }
            ],
        },
    }


def _mixed_comparison_logical_request(
    identity: str, target: str
) -> dict[str, object]:
    result = _comparison_request(identity, target)
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_mixed_comparison_logical_artifact_contract"
    graph["tensor_count"] = 4
    graph["tensors"] = [
        _tensor(1, [2, 3], [4, 1]),
        _tensor(2, [2, 3], [5, 1]),
        _tensor(3, [2, 3], [6, 1], data_type="boolean", virtual=True),
        _tensor(4, [2, 3], [7, 1], data_type="boolean"),
    ]
    graph["node_count"] = 2
    first = graph["nodes"][0]
    assert isinstance(first, dict)
    first["outputs"] = [{"name": "output", "uid": 3}]
    graph["nodes"].append(
        {
            "id": 1,
            "type": "logical_not",
            "compute_data_type": "boolean",
            "inputs": [{"name": "input", "uid": 3}],
            "outputs": [{"name": "output", "uid": 4}],
            "attributes": {
                "n_elements": 6,
                "mode": 24,
                "pointwise_mode": 24,
            },
        }
    )
    return result


def _binary_select_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float32",
    autotune: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_binary_select_artifact_contract",
            "tensor_count": 4,
            "tensors": [
                _tensor(
                    1,
                    [2, 1, 4],
                    [5, 0, 1],
                    data_type=storage_data_type,
                ),
                _tensor(
                    2,
                    [1, 3, 1],
                    [0, 2, 0],
                    data_type=storage_data_type,
                ),
                _tensor(3, [2, 3, 1], [4, 1, 0], data_type="boolean"),
                _tensor(
                    4,
                    [2, 3, 4],
                    [16, 5, 1],
                    data_type=storage_data_type,
                ),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "binary_select",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "a", "uid": 1},
                        {"name": "b", "uid": 2},
                        {"name": "t", "uid": 3},
                    ],
                    "outputs": [{"name": "output", "uid": 4}],
                    "attributes": {"n_elements": 24},
                }
            ],
        },
    }


def _mixed_binary_select_request(
    identity: str, target: str
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": True},
        "graph": {
            "name": "ascend_comparison_binary_select_relu_artifact_contract",
            "tensor_count": 7,
            "tensors": [
                _tensor(1, [2, 1], [1, 0]),
                _tensor(2, [1, 3], [0, 1]),
                _tensor(3, [2, 3], [4, 1], data_type="boolean", virtual=True),
                _tensor(4, [2, 1], [2, 0]),
                _tensor(5, [1, 3], [0, 1]),
                _tensor(6, [2, 3], [5, 1], virtual=True),
                _tensor(7, [2, 3], [3, 1]),
            ],
            "node_count": 3,
            "nodes": [
                {
                    "id": 0,
                    "type": "cmp_gt",
                    "compute_data_type": "boolean",
                    "inputs": [
                        {"name": "left", "uid": 1},
                        {"name": "right", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "n_elements": 6,
                        "pointwise_mode": 27,
                        "alpha": 1.0,
                    },
                },
                {
                    "id": 1,
                    "type": "binary_select",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "a", "uid": 4},
                        {"name": "b", "uid": 5},
                        {"name": "t", "uid": 3},
                    ],
                    "outputs": [{"name": "output", "uid": 6}],
                    "attributes": {"n_elements": 6},
                },
                {
                    "id": 2,
                    "type": "relu",
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 6}],
                    "outputs": [{"name": "output", "uid": 7}],
                    "attributes": {"n_elements": 6},
                },
            ],
        },
    }


def _layout_request(
    identity: str,
    target: str,
    *,
    operation: str,
    storage_data_type: str = "float32",
    autotune: bool = True,
) -> dict[str, object]:
    if operation == "reshape":
        input_dimensions = [2, 3]
        input_strides = [4, 1]
        output_dimensions = [3, 2]
        output_strides = [3, 1]
        attributes: dict[str, object] = {
            "n_elements": 6,
            "input_rank": 2,
            "output_rank": 2,
            "reshape_mode": 2,
            "input_dimensions": input_dimensions,
            "input_strides": input_strides,
            "output_dimensions": output_dimensions,
            "output_strides": output_strides,
        }
    elif operation == "transpose":
        input_dimensions = [2, 3, 4]
        input_strides = [20, 6, 1]
        output_dimensions = [3, 4, 2]
        output_strides = [10, 2, 1]
        attributes = {
            "n_elements": 24,
            "rank": 3,
            "permutation": [1, 2, 0],
            "input_dimensions": input_dimensions,
            "input_strides": input_strides,
            "output_dimensions": output_dimensions,
            "output_strides": output_strides,
        }
    elif operation == "slice":
        input_dimensions = [5, 6]
        input_strides = [8, 1]
        output_dimensions = [2, 3]
        output_strides = [4, 1]
        attributes = {
            "n_elements": 6,
            "rank": 2,
            "starts": [1, 1],
            "limits": [5, 6],
            "slice_strides": [2, 2],
            "input_dimensions": input_dimensions,
            "input_strides": input_strides,
            "output_dimensions": output_dimensions,
            "output_strides": output_strides,
        }
    else:
        raise ValueError(f"unsupported layout fixture operation: {operation}")
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"ascend_{operation}_artifact_contract",
            "tensor_count": 2,
            "tensors": [
                _tensor(
                    1,
                    input_dimensions,
                    input_strides,
                    data_type=storage_data_type,
                ),
                _tensor(
                    2,
                    output_dimensions,
                    output_strides,
                    data_type=storage_data_type,
                ),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 1}],
                    "outputs": [{"name": "output", "uid": 2}],
                    "attributes": attributes,
                }
            ],
        },
    }


def _reduction_request(
    identity: str,
    target: str,
    *,
    operation: str = "reduction_sum",
    storage_data_type: str = "float32",
    scalar_output: bool = False,
    padded: bool = False,
) -> dict[str, object]:
    input_dimensions = [8] if scalar_output else [2, 4, 3]
    input_strides = [1] if scalar_output else ([20, 4, 1] if padded else [12, 3, 1])
    output_dimensions = [] if scalar_output else [2, 1, 3]
    output_strides = [] if scalar_output else ([5, 5, 1] if padded else [3, 3, 1])
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": True},
        "graph": {
            "name": f"ascend_{operation}_artifact_contract",
            "tensor_count": 2,
            "tensors": [
                _tensor(1, input_dimensions, input_strides, data_type=storage_data_type),
                _tensor(2, output_dimensions, output_strides, data_type=storage_data_type),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 1}],
                    "outputs": [{"name": "output", "uid": 2}],
                    "attributes": {
                        "mode": REDUCTION_MODES[operation],
                        "outer": 1 if scalar_output else 2,
                        "reduction": 8 if scalar_output else 4,
                        "inner": 1 if scalar_output else 3,
                        "output_elements": 1 if scalar_output else 6,
                        "axis": 0 if scalar_output else -2,
                        "keep_dimensions": 0 if scalar_output else 1,
                    },
                }
            ],
        },
    }


def _matmul_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float16",
    autotune: bool = False,
    a_dimensions: list[int] | None = None,
    b_dimensions: list[int] | None = None,
    a_strides: list[int] | None = None,
    b_strides: list[int] | None = None,
    output_strides: list[int] | None = None,
) -> dict[str, object]:
    a_dimensions = [2, 17, 30] if a_dimensions is None else a_dimensions
    b_dimensions = [2, 30, 23] if b_dimensions is None else b_dimensions
    a_batch = a_dimensions[:-2]
    b_batch = b_dimensions[:-2]
    batch_rank = max(len(a_batch), len(b_batch))
    batch_dimensions = [1] * batch_rank
    for trailing in range(batch_rank):
        a_dimension = (
            a_batch[len(a_batch) - 1 - trailing]
            if trailing < len(a_batch)
            else 1
        )
        b_dimension = (
            b_batch[len(b_batch) - 1 - trailing]
            if trailing < len(b_batch)
            else 1
        )
        if a_dimension != b_dimension and a_dimension != 1 and b_dimension != 1:
            raise ValueError("test MatMul batch dimensions cannot broadcast")
        batch_dimensions[batch_rank - 1 - trailing] = max(
            a_dimension, b_dimension
        )
    output_dimensions = batch_dimensions + [a_dimensions[-2], b_dimensions[-1]]
    a_strides = (
        _contiguous_strides(a_dimensions) if a_strides is None else a_strides
    )
    b_strides = (
        _contiguous_strides(b_dimensions) if b_strides is None else b_strides
    )
    output_strides = (
        _contiguous_strides(output_dimensions)
        if output_strides is None
        else output_strides
    )
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_matmul_artifact_contract",
            "tensor_count": 3,
            "tensors": [
                _tensor(
                    1,
                    a_dimensions,
                    a_strides,
                    data_type=storage_data_type,
                ),
                _tensor(
                    2,
                    b_dimensions,
                    b_strides,
                    data_type=storage_data_type,
                ),
                _tensor(
                    3,
                    output_dimensions,
                    output_strides,
                    data_type=storage_data_type,
                ),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "matmul",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "a", "uid": 1},
                        {"name": "b", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "batch": math.prod(batch_dimensions),
                        "m": a_dimensions[-2],
                        "n": b_dimensions[-1],
                        "k": a_dimensions[-1],
                    },
                }
            ],
        },
    }


def _convolution_fprop_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float32",
    autotune: bool = False,
    input_dimensions: list[int] | None = None,
    filter_dimensions: list[int] | None = None,
    input_strides: list[int] | None = None,
    filter_strides: list[int] | None = None,
    output_strides: list[int] | None = None,
    pre_padding: list[int] | None = None,
    post_padding: list[int] | None = None,
    stride: list[int] | None = None,
    dilation: list[int] | None = None,
    groups: int = 1,
) -> dict[str, object]:
    input_dimensions = input_dimensions or [1, 2, 5, 5]
    filter_dimensions = filter_dimensions or [2, 2, 3, 3]
    spatial_rank = len(input_dimensions) - 2
    pre_padding = pre_padding or [1] * spatial_rank
    post_padding = post_padding or [1] * spatial_rank
    stride = stride or [1] * spatial_rank
    dilation = dilation or [1] * spatial_rank
    output_dimensions = [input_dimensions[0], filter_dimensions[0]]
    for axis in range(spatial_rank):
        effective_filter = dilation[axis] * (
            filter_dimensions[axis + 2] - 1
        ) + 1
        output_dimensions.append(
            (
                input_dimensions[axis + 2]
                + pre_padding[axis]
                + post_padding[axis]
                - effective_filter
            )
            // stride[axis]
            + 1
        )
    input_strides = input_strides or _contiguous_strides(input_dimensions)
    filter_strides = filter_strides or _contiguous_strides(filter_dimensions)
    output_strides = output_strides or _contiguous_strides(output_dimensions)
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_convolution_fprop_artifact_contract",
            "tensor_count": 3,
            "tensors": [
                _tensor(
                    1,
                    input_dimensions,
                    input_strides,
                    data_type=storage_data_type,
                ),
                _tensor(
                    2,
                    filter_dimensions,
                    filter_strides,
                    data_type=storage_data_type,
                ),
                _tensor(
                    3,
                    output_dimensions,
                    output_strides,
                    data_type=storage_data_type,
                ),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "convolution_fprop",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "input", "uid": 1},
                        {"name": "filter", "uid": 2},
                    ],
                    "outputs": [{"name": "output", "uid": 3}],
                    "attributes": {
                        "spatial_rank": spatial_rank,
                        "groups": groups,
                        "n_outputs": math.prod(output_dimensions),
                        "pre_padding": pre_padding,
                        "post_padding": post_padding,
                        "stride": stride,
                        "dilation": dilation,
                    },
                }
            ],
        },
    }


def _conv_bias_relu_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float16",
    autotune: bool = True,
) -> dict[str, object]:
    result = _convolution_fprop_request(
        identity,
        target,
        storage_data_type=storage_data_type,
        input_dimensions=[1, 2, 5, 5],
        filter_dimensions=[3, 2, 3, 3],
        autotune=autotune,
    )
    graph = result["graph"]
    assert isinstance(graph, dict)
    tensors = graph["tensors"]
    nodes = graph["nodes"]
    assert isinstance(tensors, list) and isinstance(nodes, list)
    convolution_output = tensors[2]
    assert isinstance(convolution_output, dict)
    convolution_output["virtual"] = True
    output_dimensions = list(convolution_output["dimensions"])
    output_strides = list(convolution_output["strides"])
    output_channels = int(output_dimensions[1])
    bias_dimensions = [1, output_channels, 1, 1]
    tensors.extend(
        [
            _tensor(
                4,
                bias_dimensions,
                _contiguous_strides(bias_dimensions),
                data_type=storage_data_type,
            ),
            _tensor(
                5,
                output_dimensions,
                output_strides,
                data_type=storage_data_type,
                virtual=True,
            ),
            _tensor(
                6,
                output_dimensions,
                output_strides,
                data_type=storage_data_type,
            ),
        ]
    )
    n_elements = math.prod(output_dimensions)
    nodes.extend(
        [
            {
                "id": 1,
                "type": "add",
                "compute_data_type": "float32",
                "inputs": [
                    {"name": "left", "uid": 3},
                    {"name": "right", "uid": 4},
                ],
                "outputs": [{"name": "output", "uid": 5}],
                "attributes": {
                    "n_elements": n_elements,
                    "pointwise_mode": POINTWISE_MODES["add"],
                    "alpha": 1.0,
                },
            },
            {
                "id": 2,
                "type": "relu",
                "compute_data_type": "float32",
                "inputs": [{"name": "input", "uid": 5}],
                "outputs": [{"name": "output", "uid": 6}],
                "attributes": {
                    "mode": UNARY_POINTWISE_MODES["relu"],
                    "n_elements": n_elements,
                    "negative_slope": 0.0,
                    "lower_clip": 0.0,
                    "upper_clip": 0.0,
                    "has_upper_clip": 0,
                    "relu_lower_clip_slope": 0.0,
                    "relu_lower_clip": 0.0,
                    "relu_upper_clip": 0.0,
                    "relu_upper_clip_set": False,
                    "swish_beta": 1.0,
                    "elu_alpha": 1.0,
                    "softplus_beta": 1.0,
                },
            },
        ]
    )
    graph["name"] = "ascend_conv_bias_relu_artifact_contract"
    graph["tensor_count"] = 6
    graph["node_count"] = 3
    return result


def _batchnorm_training_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float16",
    autotune: bool = False,
) -> dict[str, object]:
    dimensions = [2, 3, 2, 2]
    strides = [12, 4, 2, 1]
    parameter_dimensions = [1, 3, 1, 1]
    parameter_strides = [3, 1, 1, 1]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_batchnorm_training_artifact_contract",
            "tensor_count": 10,
            "tensors": [
                _tensor(1, dimensions, strides, data_type=storage_data_type),
                _tensor(2, parameter_dimensions, parameter_strides,
                        data_type=storage_data_type),
                _tensor(3, parameter_dimensions, parameter_strides,
                        data_type=storage_data_type),
                _tensor(4, parameter_dimensions, parameter_strides),
                _tensor(5, parameter_dimensions, parameter_strides),
                _tensor(6, dimensions, strides, data_type=storage_data_type),
                _tensor(7, parameter_dimensions, parameter_strides),
                _tensor(8, parameter_dimensions, parameter_strides),
                _tensor(9, parameter_dimensions, parameter_strides),
                _tensor(10, parameter_dimensions, parameter_strides),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "batchnorm",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "x", "uid": 1},
                        {"name": "scale", "uid": 2},
                        {"name": "bias", "uid": 3},
                        {"name": "previous_running_mean", "uid": 4},
                        {"name": "previous_running_variance", "uid": 5},
                    ],
                    "outputs": [
                        {"name": "y", "uid": 6},
                        {"name": "mean", "uid": 7},
                        {"name": "inv_variance", "uid": 8},
                        {"name": "next_running_mean", "uid": 9},
                        {"name": "next_running_variance", "uid": 10},
                    ],
                    "attributes": {
                        "n_elements": 24,
                        "batch": 2,
                        "channels": 3,
                        "spatial": 4,
                        "rank": 4,
                        "epsilon": 1.0e-3,
                        "momentum": 0.1,
                        "dimensions": dimensions,
                        "x_strides": strides,
                        "y_strides": strides,
                    },
                }
            ],
        },
    }


def _batchnorm_inference_request(
    identity: str,
    target: str,
    *,
    storage_data_type: str = "float16",
    dimensions: list[int] | None = None,
    x_strides: list[int] | None = None,
    y_strides: list[int] | None = None,
) -> dict[str, object]:
    dimensions = [2, 3] if dimensions is None else dimensions
    x_strides = [3, 1] if x_strides is None else x_strides
    y_strides = [3, 1] if y_strides is None else y_strides
    channels = dimensions[1]
    spatial = math.prod(dimensions[2:])
    n_elements = math.prod(dimensions)
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": False},
        "graph": {
            "name": "ascend_batchnorm_inference_artifact_contract",
            "tensor_count": 6,
            "tensors": [
                _tensor(1, dimensions, x_strides, data_type=storage_data_type),
                _tensor(2, [channels], [1]),
                _tensor(3, [channels], [1]),
                _tensor(4, [channels], [1]),
                _tensor(5, [channels], [1]),
                _tensor(6, dimensions, y_strides, data_type=storage_data_type),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "batchnorm_inference",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "x", "uid": 1},
                        {"name": "mean", "uid": 2},
                        {"name": "inv_variance", "uid": 3},
                        {"name": "scale", "uid": 4},
                        {"name": "bias", "uid": 5},
                    ],
                    "outputs": [{"name": "y", "uid": 6}],
                    "attributes": {
                        "n_elements": n_elements,
                        "channels": channels,
                        "spatial": spatial,
                        "rank": len(dimensions),
                        "dimensions": dimensions,
                        "x_strides": x_strides,
                        "y_strides": y_strides,
                    },
                }
            ],
        },
    }


def _rmsnorm_request(identity: str, target: str) -> dict[str, object]:
    dimensions = [2, 5, 17]
    strides = [85, 17, 1]
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": False},
        "graph": {
            "name": "ascend_rmsnorm_artifact_contract",
            "tensor_count": 5,
            "tensors": [
                _tensor(1, dimensions, strides, data_type="float16"),
                _tensor(2, [17], [1], data_type="float16"),
                _tensor(3, [17], [1], data_type="float16"),
                _tensor(4, dimensions, strides, data_type="float16"),
                _tensor(5, [2, 5, 1], [5, 1, 1]),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "rmsnorm",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "x", "uid": 1},
                        {"name": "scale", "uid": 2},
                        {"name": "bias", "uid": 3},
                    ],
                    "outputs": [
                        {"name": "y", "uid": 4},
                        {"name": "inv_variance", "uid": 5},
                    ],
                    "attributes": {
                        "rows": 10,
                        "normalized_elements": 17,
                        "epsilon": 1.0e-5,
                        "forward_phase": 2,
                    },
                }
            ],
        },
    }


def _layernorm_request(
    identity: str,
    target: str,
    *,
    data_type: str = "float16",
    dimensions: list[int] | None = None,
    normalized_rank: int = 1,
    virtual_y: bool = False,
    autotune: bool = False,
) -> dict[str, object]:
    dimensions = [2, 5, 17] if dimensions is None else dimensions
    if normalized_rank < 1 or normalized_rank > len(dimensions):
        raise ValueError("normalized_rank is outside LayerNorm input rank")
    strides = _contiguous_strides(dimensions)
    leading_rank = len(dimensions) - normalized_rank
    parameter_dimensions = [1] * leading_rank + dimensions[-normalized_rank:]
    parameter_strides = _contiguous_strides(parameter_dimensions)
    statistic_dimensions = dimensions[:leading_rank] + [1] * normalized_rank
    statistic_strides = _contiguous_strides(statistic_dimensions)
    normalized_elements = math.prod(dimensions[-normalized_rank:])
    rows = math.prod(dimensions) // normalized_elements
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": "ascend_layernorm_artifact_contract",
            "tensor_count": 6,
            "tensors": [
                _tensor(1, dimensions, strides, data_type=data_type),
                _tensor(
                    2,
                    parameter_dimensions,
                    parameter_strides,
                    data_type=data_type,
                ),
                _tensor(
                    3,
                    parameter_dimensions,
                    parameter_strides,
                    data_type=data_type,
                ),
                _tensor(
                    4,
                    dimensions,
                    strides,
                    data_type=data_type,
                    virtual=virtual_y,
                ),
                _tensor(5, statistic_dimensions, statistic_strides),
                _tensor(6, statistic_dimensions, statistic_strides),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": "layernorm",
                    "compute_data_type": "float32",
                    "inputs": [
                        {"name": "x", "uid": 1},
                        {"name": "scale", "uid": 2},
                        {"name": "bias", "uid": 3},
                    ],
                    "outputs": [
                        {"name": "y", "uid": 4},
                        {"name": "mean", "uid": 5},
                        {"name": "inv_variance", "uid": 6},
                    ],
                    "attributes": {
                        "rows": rows,
                        "normalized_elements": normalized_elements,
                        "epsilon": 1.0e-5,
                        "forward_phase": 2,
                    },
                }
            ],
        },
    }


def _batchnorm_virtual_dag_request(
    identity: str, target: str
) -> dict[str, object]:
    result = _batchnorm_inference_request(identity, target)
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_batchnorm_inference_virtual_dag_contract"
    graph["tensor_count"] = 8
    graph["tensors"] = [
        _tensor(1, [2, 3], [3, 1], data_type="float16"),
        _tensor(2, [2, 3], [3, 1], data_type="float16", virtual=True),
        _tensor(3, [3], [1]),
        _tensor(4, [3], [1]),
        _tensor(5, [3], [1]),
        _tensor(6, [3], [1]),
        _tensor(7, [2, 3], [3, 1], data_type="float16", virtual=True),
        _tensor(8, [2, 3], [3, 1], data_type="float16"),
    ]
    graph["node_count"] = 3
    graph["nodes"] = [
        {
            "id": 0,
            "type": "identity",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 1}],
            "outputs": [{"name": "output", "uid": 2}],
            "attributes": {"n_elements": 6},
        },
        {
            "id": 1,
            "type": "batchnorm_inference",
            "compute_data_type": "float32",
            "inputs": [
                {"name": "x", "uid": 2},
                {"name": "mean", "uid": 3},
                {"name": "inv_variance", "uid": 4},
                {"name": "scale", "uid": 5},
                {"name": "bias", "uid": 6},
            ],
            "outputs": [{"name": "y", "uid": 7}],
            "attributes": {
                "n_elements": 6,
                "channels": 3,
                "spatial": 1,
                "rank": 2,
                "dimensions": [2, 3],
                "x_strides": [3, 1],
                "y_strides": [3, 1],
            },
        },
        {
            "id": 2,
            "type": "relu",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 7}],
            "outputs": [{"name": "output", "uid": 8}],
            "attributes": {"n_elements": 6},
        },
    ]
    return result


def _mixed_reduction_layout_pointwise_request(
    identity: str, target: str
) -> dict[str, object]:
    result = _reduction_request(identity, target, operation="reduction_avg")
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_reduction_layout_pointwise_virtual_contract"
    graph["tensor_count"] = 4
    graph["tensors"] = [
        _tensor(1, [2, 4, 3], [12, 3, 1]),
        _tensor(2, [2, 1, 3], [3, 3, 1], virtual=True),
        _tensor(3, [2, 3], [3, 1], virtual=True),
        _tensor(4, [2, 3], [3, 1]),
    ]
    reduction = graph["nodes"][0]
    assert isinstance(reduction, dict)
    reduction["outputs"] = [{"name": "output", "uid": 2}]
    graph["node_count"] = 3
    graph["nodes"] = [
        reduction,
        {
            "id": 1,
            "type": "reshape",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 2}],
            "outputs": [{"name": "output", "uid": 3}],
            "attributes": {
                "n_elements": 6,
                "input_rank": 3,
                "output_rank": 2,
                "reshape_mode": 2,
                "input_dimensions": [2, 1, 3],
                "input_strides": [3, 3, 1],
                "output_dimensions": [2, 3],
                "output_strides": [3, 1],
            },
        },
        {
            "id": 2,
            "type": "relu",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 3}],
            "outputs": [{"name": "output", "uid": 4}],
            "attributes": {"n_elements": 6},
        },
    ]
    return result


def _optimized_layout_request(
    identity: str, target: str, operation: str
) -> dict[str, object]:
    result = _layout_request(identity, target, operation=operation)
    graph = result["graph"]
    assert isinstance(graph, dict)
    node = graph["nodes"][0]
    assert isinstance(node, dict)
    attributes = node["attributes"]
    assert isinstance(attributes, dict)
    fixtures = {
        "reshape": ([2, 3], [3, 1], [3, 2], [2, 1]),
        "transpose": ([2, 3, 4], [12, 4, 1], [3, 4, 2], [4, 1, 12]),
        "slice": ([5, 6], [6, 1], [2, 3], [12, 2]),
    }
    input_dims, input_strides, output_dims, output_strides = fixtures[operation]
    graph["tensors"] = [
        _tensor(1, input_dims, input_strides),
        _tensor(2, output_dims, output_strides),
    ]
    attributes.update(
        input_dimensions=input_dims,
        input_strides=input_strides,
        output_dimensions=output_dims,
        output_strides=output_strides,
    )
    return result


def _scalar_reshape_request(
    identity: str, target: str
) -> dict[str, object]:
    result = _layout_request(
        identity, target, operation="reshape", autotune=False
    )
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["tensors"] = [_tensor(1, [], []), _tensor(2, [], [])]
    node = graph["nodes"][0]
    assert isinstance(node, dict)
    node["attributes"] = {
        "n_elements": 1,
        "input_rank": 0,
        "output_rank": 0,
        "reshape_mode": 2,
        "input_dimensions": [],
        "input_strides": [],
        "output_dimensions": [],
        "output_strides": [],
    }
    return result


def _mixed_layout_pointwise_request(
    identity: str, target: str
) -> dict[str, object]:
    result = _layout_request(identity, target, operation="reshape")
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_layout_pointwise_virtual_artifact_contract"
    graph["tensor_count"] = 6
    graph["tensors"] = [
        _tensor(1, [2, 3], [4, 1]),
        _tensor(2, [3, 2], [2, 1], virtual=True),
        _tensor(3, [3, 2], [2, 1], virtual=True),
        _tensor(4, [2, 3], [4, 1], virtual=True),
        _tensor(5, [2, 2], [2, 1], virtual=True),
        _tensor(6, [2, 2], [2, 1]),
    ]
    graph["node_count"] = 5
    graph["nodes"] = [
        {
            "id": 0,
            "type": "reshape",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 1}],
            "outputs": [{"name": "output", "uid": 2}],
            "attributes": {
                "n_elements": 6,
                "input_rank": 2,
                "output_rank": 2,
                "reshape_mode": 2,
                "input_dimensions": [2, 3],
                "input_strides": [4, 1],
                "output_dimensions": [3, 2],
                "output_strides": [2, 1],
            },
        },
        {
            "id": 1,
            "type": "relu",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 2}],
            "outputs": [{"name": "output", "uid": 3}],
            "attributes": {"n_elements": 6},
        },
        {
            "id": 2,
            "type": "transpose",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 3}],
            "outputs": [{"name": "output", "uid": 4}],
            "attributes": {
                "n_elements": 6,
                "rank": 2,
                "permutation": [1, 0],
                "input_dimensions": [3, 2],
                "input_strides": [2, 1],
                "output_dimensions": [2, 3],
                "output_strides": [4, 1],
            },
        },
        {
            "id": 3,
            "type": "slice",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 4}],
            "outputs": [{"name": "output", "uid": 5}],
            "attributes": {
                "n_elements": 4,
                "rank": 2,
                "starts": [0, 1],
                "limits": [2, 3],
                "slice_strides": [1, 1],
                "input_dimensions": [2, 3],
                "input_strides": [4, 1],
                "output_dimensions": [2, 2],
                "output_strides": [2, 1],
            },
        },
        {
            "id": 4,
            "type": "identity",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 5}],
            "outputs": [{"name": "output", "uid": 6}],
            "attributes": {"n_elements": 4},
        },
    ]
    return result


def _unary_request(
    identity: str,
    target: str,
    *,
    operation: str = "relu",
    storage_data_type: str = "float32",
    autotune: bool = True,
    negative_slope: float = 0.2,
    lower_clip: float = -0.5,
    upper_clip: float = 3.0,
    has_upper_clip: int = 1,
    swish_beta: float = 1.0,
    elu_alpha: float = 1.0,
    softplus_beta: float = 1.0,
) -> dict[str, object]:
    if operation != "relu":
        negative_slope = 0.0
        lower_clip = 0.0
        upper_clip = 0.0
        has_upper_clip = 0
    return {
        "schema_version": 3,
        "flagdnn_version": "0.2.0",
        "backend": "ascend",
        "target": target,
        "compiler_identity": identity,
        "build_options": {"heuristic_modes": ["A"], "autotune": autotune},
        "graph": {
            "name": f"ascend_{operation}_artifact_contract",
            "tensor_count": 2,
            "tensors": [
                _tensor(1, [2, 3], [4, 1], data_type=storage_data_type),
                _tensor(2, [2, 3], [5, 1], data_type=storage_data_type),
            ],
            "node_count": 1,
            "nodes": [
                {
                    "id": 0,
                    "type": operation,
                    "compute_data_type": "float32",
                    "inputs": [{"name": "input", "uid": 1}],
                    "outputs": [{"name": "output", "uid": 2}],
                    "attributes": {
                        "mode": UNARY_POINTWISE_MODES[operation],
                        "n_elements": 6,
                        "negative_slope": negative_slope,
                        "lower_clip": lower_clip,
                        "upper_clip": upper_clip,
                        "has_upper_clip": has_upper_clip,
                        "relu_lower_clip_slope": negative_slope,
                        "relu_lower_clip": lower_clip,
                        "relu_upper_clip": upper_clip,
                        "relu_upper_clip_set": bool(has_upper_clip),
                        "swish_beta": swish_beta,
                        "elu_alpha": elu_alpha,
                        "softplus_beta": softplus_beta,
                    },
                }
            ],
        },
    }


def _mixed_pointwise_request(identity: str, target: str) -> dict[str, object]:
    result = _request(identity, target, autotune=True)
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["name"] = "ascend_mixed_binary_unary_contract"
    graph["tensor_count"] = 6
    graph["tensors"] = [
        _tensor(1, [2, 3], [3, 1]),
        _tensor(2, [2, 3], [3, 1]),
        _tensor(3, [2, 3], [3, 1], virtual=True),
        _tensor(4, [2, 3], [3, 1], virtual=True),
        _tensor(5, [2, 3], [3, 1]),
        _tensor(6, [2, 3], [3, 1]),
    ]
    graph["node_count"] = 3
    graph["nodes"] = [
        {
            "id": 0,
            "type": "add",
            "compute_data_type": "float32",
            "inputs": [
                {"name": "left", "uid": 1},
                {"name": "right", "uid": 2},
            ],
            "outputs": [{"name": "output", "uid": 3}],
            "attributes": {
                "n_elements": 6,
                "pointwise_mode": 1,
                "alpha": 1.0,
            },
        },
        {
            "id": 1,
            "type": "relu",
            "compute_data_type": "float32",
            "inputs": [{"name": "input", "uid": 3}],
            "outputs": [{"name": "output", "uid": 4}],
            "attributes": {
                "mode": 2,
                "n_elements": 6,
                "negative_slope": 0.2,
                "lower_clip": -0.5,
                "upper_clip": 3.0,
                "has_upper_clip": 1,
                "relu_lower_clip_slope": 0.2,
                "relu_lower_clip": -0.5,
                "relu_upper_clip": 3.0,
                "relu_upper_clip_set": True,
                "swish_beta": 1.0,
                "elu_alpha": 1.0,
                "softplus_beta": 1.0,
            },
        },
        {
            "id": 2,
            "type": "mul",
            "compute_data_type": "float32",
            "inputs": [
                {"name": "left", "uid": 4},
                {"name": "right", "uid": 5},
            ],
            "outputs": [{"name": "output", "uid": 6}],
            "attributes": {
                "n_elements": 6,
                "pointwise_mode": 18,
                "alpha": 1.0,
            },
        },
    ]
    return result


def _mixed_unary_math_request(
    identity: str,
    target: str,
    operations: tuple[str, ...] = ("sqrt", "rsqrt", "reciprocal"),
) -> dict[str, object]:
    if not operations:
        raise ValueError("mixed unary fixture requires at least one operation")
    result = _unary_request(
        identity, target, operation=operations[0], autotune=True
    )
    graph = result["graph"]
    assert isinstance(graph, dict)
    first_node = graph["nodes"][0]
    assert isinstance(first_node, dict)
    base_attributes = first_node["attributes"]
    assert isinstance(base_attributes, dict)
    graph["name"] = "ascend_mixed_" + "_".join(operations) + "_contract"
    graph["tensor_count"] = len(operations) + 1
    graph["tensors"] = [
        _tensor(
            uid,
            [2, 3],
            [3, 1],
            virtual=1 < uid <= len(operations),
        )
        for uid in range(1, len(operations) + 2)
    ]
    graph["node_count"] = len(operations)
    graph["nodes"] = []
    for node_id, operation in enumerate(operations):
        attributes = copy.deepcopy(base_attributes)
        attributes["mode"] = UNARY_POINTWISE_MODES[operation]
        graph["nodes"].append(
            {
                "id": node_id,
                "type": operation,
                "compute_data_type": "float32",
                "inputs": [{"name": "input", "uid": node_id + 1}],
                "outputs": [{"name": "output", "uid": node_id + 2}],
                "attributes": attributes,
            }
        )
    return result


def _mixed_activation_request(identity: str, target: str) -> dict[str, object]:
    result = _unary_request(
        identity, target, operation="sigmoid", autotune=True
    )
    graph = result["graph"]
    assert isinstance(graph, dict)
    first_node = graph["nodes"][0]
    assert isinstance(first_node, dict)
    base_attributes = first_node["attributes"]
    assert isinstance(base_attributes, dict)
    graph["name"] = "ascend_mixed_sigmoid_elu_softplus_swish_contract"
    graph["tensor_count"] = 5
    graph["tensors"] = [
        _tensor(1, [2, 3], [3, 1]),
        _tensor(2, [2, 3], [3, 1], virtual=True),
        _tensor(3, [2, 3], [3, 1], virtual=True),
        _tensor(4, [2, 3], [3, 1], virtual=True),
        _tensor(5, [2, 3], [3, 1]),
    ]
    specifications = (
        ("sigmoid", 1, 2, {}),
        ("elu", 2, 3, {"elu_alpha": 1.25}),
        ("softplus", 3, 4, {"softplus_beta": 0.75}),
        ("swish", 4, 5, {"swish_beta": 0.5}),
    )
    graph["node_count"] = len(specifications)
    graph["nodes"] = []
    for node_id, (
        operation,
        input_uid,
        output_uid,
        semantic_attrs,
    ) in enumerate(specifications):
        attributes = copy.deepcopy(base_attributes)
        attributes["mode"] = UNARY_POINTWISE_MODES[operation]
        attributes.update(semantic_attrs)
        graph["nodes"].append(
            {
                "id": node_id,
                "type": operation,
                "compute_data_type": "float32",
                "inputs": [{"name": "input", "uid": input_uid}],
                "outputs": [{"name": "output", "uid": output_uid}],
                "attributes": attributes,
            }
        )
    return result


def _mixed_gelu_request(identity: str, target: str) -> dict[str, object]:
    result = _unary_request(
        identity, target, operation="gelu", autotune=True
    )
    graph = result["graph"]
    assert isinstance(graph, dict)
    first_node = graph["nodes"][0]
    assert isinstance(first_node, dict)
    base_attributes = first_node["attributes"]
    assert isinstance(base_attributes, dict)
    graph["name"] = "ascend_mixed_gelu_gelu_approx_tanh_contract"
    graph["tensor_count"] = 3
    graph["tensors"] = [
        _tensor(1, [2, 3], [3, 1]),
        _tensor(2, [2, 3], [3, 1], virtual=True),
        _tensor(3, [2, 3], [3, 1]),
    ]
    graph["node_count"] = 2
    graph["nodes"] = []
    for node_id, (operation, input_uid, output_uid) in enumerate(
        (("gelu", 1, 2), ("gelu_approx_tanh", 2, 3))
    ):
        attributes = copy.deepcopy(base_attributes)
        attributes["mode"] = UNARY_POINTWISE_MODES[operation]
        graph["nodes"].append(
            {
                "id": node_id,
                "type": operation,
                "compute_data_type": "float32",
                "inputs": [{"name": "input", "uid": input_uid}],
                "outputs": [{"name": "output", "uid": output_uid}],
                "attributes": attributes,
            }
        )
    return result


def _large_channels_last_dense_request(
    identity: str, target: str
) -> dict[str, object]:
    dimensions = [4, 16, 64, 128]
    strides = [131072, 1, 2048, 16]
    result = _request(identity, target)
    graph = result["graph"]
    assert isinstance(graph, dict)
    graph["tensor_count"] = 3
    graph["tensors"] = [_tensor(uid, dimensions, strides) for uid in (1, 2, 3)]
    graph["node_count"] = 1
    graph["nodes"] = [
        {
            "id": 0,
            "type": "add",
            "compute_data_type": "float32",
            "inputs": [
                {"name": "left", "uid": 1},
                {"name": "right", "uid": 2},
            ],
            "outputs": [{"name": "output", "uid": 3}],
            "attributes": {
                "n_elements": 4 * 16 * 64 * 128,
                "pointwise_mode": 1,
                "alpha": 1.0,
            },
        }
    ]
    return result


def _write_request(path: Path, document: dict[str, object]) -> str:
    contents = _canonical_request(document)
    path.write_bytes(contents)
    return hashlib.sha256(contents).hexdigest()


def _payload(manifest: dict[str, Any], stage: int = 0) -> dict[str, Any]:
    return manifest["program"]["stages"][stage]["candidates"][0]["payload"]


def _change_signature_meta(
    manifest: dict[str, Any], name: str, token_index: int
) -> None:
    payload = _payload(manifest)
    value = int(payload["meta"][name]) + 1
    payload["meta"][name] = value
    tokens = payload["full_signature"].split(",")
    tokens[token_index] = str(value)
    payload["full_signature"] = ",".join(tokens)


def _set_stage_meta(
    manifest: dict[str, Any], stage_index: int, name: str, value: int | float
) -> None:
    payload = _payload(manifest, stage_index)
    entry_point = payload["entry_point"]
    if entry_point == "binary_contiguous_kernel":
        meta_names = ["OP_KIND", "ALPHA", "BLOCK_SIZE", "WORKER_COUNT"]
        runtime_count = 4
    elif entry_point == "binary_strided_kernel":
        meta_names = [
            f"{prefix}_{axis}"
            for prefix in (
                "DIM",
                "LEFT_STRIDE",
                "RIGHT_STRIDE",
                "OUTPUT_STRIDE",
            )
            for axis in range(8)
        ] + ["OP_KIND", "ALPHA", "BLOCK_SIZE", "WORKER_COUNT"]
        runtime_count = 4
    elif entry_point == "unary_pointwise_contiguous_kernel":
        meta_names = [
            "OPERATION",
            "negative_slope",
            "lower_clip",
            "upper_clip",
            "HAS_UPPER_CLIP",
            "SWISH_BETA",
            "ELU_ALPHA",
            "SOFTPLUS_BETA",
            "BLOCK_SIZE",
            "WORKER_COUNT",
        ]
        runtime_count = 3
    elif entry_point == "layout_copy_kernel":
        meta_names = (
            ["INPUT_BASE", "ELEMENT_SIZE_BYTES", "LAYOUT_MODE"]
            + [f"INPUT_DIM_{axis}" for axis in range(8)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(8)]
            + [f"OUTPUT_DIM_{axis}" for axis in range(8)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(8)]
            + ["BLOCK_SIZE", "WORKER_COUNT"]
        )
        runtime_count = 3
    else:
        meta_names = [
            f"{prefix}_{axis}"
            for prefix in ("DIM", "INPUT_STRIDE", "OUTPUT_STRIDE")
            for axis in range(8)
        ] + [
            "OPERATION",
            "negative_slope",
            "lower_clip",
            "upper_clip",
            "HAS_UPPER_CLIP",
            "SWISH_BETA",
            "ELU_ALPHA",
            "SOFTPLUS_BETA",
            "BLOCK_SIZE",
            "WORKER_COUNT",
        ]
        runtime_count = 3
    payload["meta"][name] = value
    tokens = payload["full_signature"].split(",")
    tokens[runtime_count + meta_names.index(name)] = str(value)
    payload["full_signature"] = ",".join(tokens)


def _mutations(
    target: str,
) -> list[tuple[str, Callable[[dict[str, Any]], None]]]:
    target_prefix, separator, encoded_ai_core_count = target.rpartition(
        "_aic_"
    )
    if not separator:
        raise RuntimeError("mutation target has no _aic_ marker")
    other_ai_core_count = "1" if encoded_ai_core_count != "1" else "24"

    def manifest_target(manifest: dict[str, Any]) -> None:
        manifest["target"] = f"{target_prefix}_aic_{other_ai_core_count}"

    def compiler_identity(manifest: dict[str, Any]) -> None:
        manifest["compiler"]["identity_sha256"] = "0" * 64

    def n_elements(manifest: dict[str, Any]) -> None:
        manifest["program"]["stages"][0]["argument_sources"][3]["value"] = 7

    def output_uid(manifest: dict[str, Any]) -> None:
        manifest["program"]["stages"][0]["argument_sources"][2]["uid"] = 5

    def dtype_signature(manifest: dict[str, Any]) -> None:
        payload = _payload(manifest)
        tokens = payload["full_signature"].split(",")
        tokens[0] = "*fp16:16"
        payload["full_signature"] = ",".join(tokens)

    def strided_dimension(manifest: dict[str, Any]) -> None:
        _change_signature_meta(manifest, "DIM_7", 4 + 7)

    def strided_stride(manifest: dict[str, Any]) -> None:
        _change_signature_meta(manifest, "LEFT_STRIDE_7", 4 + 8 + 7)

    def launch_grid(manifest: dict[str, Any]) -> None:
        _payload(manifest)["grid"] = [2, 1, 1]

    def stage_workspace(manifest: dict[str, Any]) -> None:
        workspace = manifest["program"]["stages"][0]["workspace"]
        workspace["size"] = int(workspace["size"]) + 4

    def dependency(manifest: dict[str, Any]) -> None:
        manifest["program"]["stages"][1]["dependencies"] = []

    def worker_count(manifest: dict[str, Any]) -> None:
        payload = _payload(manifest)
        token_index = len(payload["full_signature"].split(",")) - 1
        _change_signature_meta(manifest, "WORKER_COUNT", token_index)

    def source_ownership(manifest: dict[str, Any]) -> None:
        kernel = manifest["program"]["stages"][0]["kernel"]
        kernel["provider"] = "common_triton"
        kernel["ownership"] = "common"

    return [
        ("manifest_target", manifest_target),
        ("compiler_identity", compiler_identity),
        ("n_elements", n_elements),
        ("output_uid", output_uid),
        ("dtype_signature", dtype_signature),
        ("strided_dimension", strided_dimension),
        ("strided_stride", strided_stride),
        ("launch_grid", launch_grid),
        ("stage_workspace", stage_workspace),
        ("dependency", dependency),
        ("worker_count", worker_count),
        ("source_ownership", source_ownership),
    ]


def _run_driver(
    driver: Path,
    target: str,
    request: Path,
    artifact: Path,
    expectation: str,
    case_name: str,
    *,
    ai_core_count: int | None = None,
) -> None:
    expected_ai_core_count = (
        aicore_count_from_target(target)
        if ai_core_count is None
        else ai_core_count
    )
    completed = subprocess.run(
        [
            str(driver),
            target,
            str(expected_ai_core_count),
            str(request),
            str(artifact),
            expectation,
            case_name,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"artifact case {case_name} failed: {completed.stderr.strip()}"
        )


def main() -> int:
    if len(sys.argv) != 3:
        raise RuntimeError("usage: artifact_contract_test.py DRIVER TARGET")
    driver = Path(sys.argv[1]).resolve(strict=True)
    target = sys.argv[2]
    provider = get_provider("ascend")
    identity = provider.compiler_identity(target, "libtriton_jit")
    target_prefix, separator, _ = target.rpartition("_aic_")
    if not separator:
        raise RuntimeError("artifact contract target has no _aic_ marker")
    target_head, cann_separator, target_cann = target_prefix.partition(
        "_cann_"
    )
    if not cann_separator or not target_head.startswith("ascend_"):
        raise RuntimeError("artifact contract target has no Ascend SoC")
    target_soc = target_head.removeprefix("ascend_")

    with tempfile.TemporaryDirectory(
        prefix="flagdnn-ascend-artifact-"
    ) as root:
        fixture_root = Path(root)
        valid_fixtures: dict[str, tuple[Path, Path]] = {}
        for storage_data_type in ("float32", "float16", "bfloat16"):
            request_path = fixture_root / f"request-{storage_data_type}.json"
            baseline = fixture_root / f"baseline-{storage_data_type}"
            _write_request(
                request_path,
                _request(
                    identity["identity_sha256"],
                    target,
                    storage_data_type=storage_data_type,
                ),
            )
            provider.compile_request(request_path, baseline, "libtriton_jit")
            valid_fixtures[storage_data_type] = (request_path, baseline)
            _run_driver(
                driver,
                target,
                request_path,
                baseline,
                "success",
                f"valid_{storage_data_type}_storage_float32_compute",
            )

        request_path, baseline = valid_fixtures["float32"]

        convolution_request = fixture_root / "request-convolution-fprop-fp16.json"
        convolution_baseline = fixture_root / "baseline-convolution-fprop-fp16"
        convolution_grouped_options = {
            "storage_data_type": "float16",
            "input_dimensions": [1, 4, 7, 9],
            "filter_dimensions": [6, 2, 3, 2],
            "input_strides": [500, 100, 12, 1],
            "filter_strides": [100, 30, 8, 1],
            "output_strides": [400, 60, 10, 1],
            "pre_padding": [1, 0],
            "post_padding": [2, 1],
            "stride": [2, 1],
            "dilation": [1, 2],
            "groups": 2,
        }
        convolution_request_document = _convolution_fprop_request(
            identity["identity_sha256"], target, **convolution_grouped_options
        )
        _write_request(convolution_request, convolution_request_document)
        provider.compile_request(
            convolution_request, convolution_baseline, "libtriton_jit"
        )
        convolution_manifest = json.loads(
            (convolution_baseline / "manifest.json").read_bytes()
        )
        convolution_stage = convolution_manifest["program"]["stages"][0]
        convolution_payload = convolution_stage["candidates"][0]["payload"]
        assert convolution_stage["kernel_family"] == "convolution_fprop"
        assert [
            argument["name"] for argument in convolution_stage["argument_sources"]
        ] == ["input_ptr", "filter_ptr", "output_ptr", "n_elements"]
        assert (
            convolution_payload["entry_point"]
            == "convolution_fprop_persistent_kernel"
        )
        assert convolution_payload["meta"]["SPATIAL_RANK"] == 2
        assert convolution_payload["meta"]["GROUPS"] == 2
        assert convolution_payload["meta"]["BLOCK_SIZE"] == 256
        _run_driver(
            driver,
            target,
            convolution_request,
            convolution_baseline,
            "success",
            "valid_convolution_fprop_fp16_grouped_fixed",
        )
        convolution_autotune_request = (
            fixture_root / "request-convolution-fprop-fp16-autotune.json"
        )
        convolution_autotune_baseline = (
            fixture_root / "baseline-convolution-fprop-fp16-autotune"
        )
        _write_request(
            convolution_autotune_request,
            _convolution_fprop_request(
                identity["identity_sha256"],
                target,
                autotune=True,
                **convolution_grouped_options,
            ),
        )
        provider.compile_request(
            convolution_autotune_request,
            convolution_autotune_baseline,
            "libtriton_jit",
        )
        convolution_autotune_manifest = json.loads(
            (convolution_autotune_baseline / "manifest.json").read_bytes()
        )
        assert len(
            convolution_autotune_manifest["program"]["stages"][0]["candidates"]
        ) == 2
        _run_driver(
            driver,
            target,
            convolution_autotune_request,
            convolution_autotune_baseline,
            "success",
            "valid_convolution_fprop_fp16_grouped_autotuned",
        )
        convolution_rank_cases = {
            "valid_convolution_fprop_fp32_rank1_channels_last_fixed":
                _convolution_fprop_request(
                    identity["identity_sha256"],
                    target,
                    input_dimensions=[2, 4, 16],
                    filter_dimensions=[6, 4, 3],
                    output_strides=[96, 1, 6],
                ),
            "valid_convolution_fprop_bf16_rank3_channels_last_fixed":
                _convolution_fprop_request(
                    identity["identity_sha256"],
                    target,
                    storage_data_type="bfloat16",
                    input_dimensions=[1, 2, 5, 6, 7],
                    filter_dimensions=[4, 2, 3, 3, 3],
                    input_strides=[420, 1, 84, 14, 2],
                    filter_strides=[54, 1, 18, 6, 2],
                    output_strides=[840, 1, 168, 28, 4],
                ),
        }
        for case_name, request_document in convolution_rank_cases.items():
            rank_request = fixture_root / f"request-{case_name}.json"
            rank_baseline = fixture_root / f"baseline-{case_name}"
            _write_request(rank_request, request_document)
            provider.compile_request(rank_request, rank_baseline, "libtriton_jit")
            rank_manifest = json.loads((rank_baseline / "manifest.json").read_bytes())
            rank_stage = rank_manifest["program"]["stages"][0]
            rank = request_document["graph"]["nodes"][0]["attributes"][
                "spatial_rank"
            ]
            assert rank_stage["candidates"][0]["payload"]["meta"][
                "SPATIAL_RANK"
            ] == rank
            _run_driver(
                driver,
                target,
                rank_request,
                rank_baseline,
                "success",
                case_name,
            )

        conv_bias_relu_request = fixture_root / "request-conv-bias-relu.json"
        conv_bias_relu_baseline = fixture_root / "baseline-conv-bias-relu"
        conv_bias_relu_document = _conv_bias_relu_request(
            identity["identity_sha256"], target
        )
        _write_request(conv_bias_relu_request, conv_bias_relu_document)
        provider.compile_request(
            conv_bias_relu_request,
            conv_bias_relu_baseline,
            "libtriton_jit",
        )
        conv_bias_relu_manifest = json.loads(
            (conv_bias_relu_baseline / "manifest.json").read_bytes()
        )
        assert [
            stage["kernel_family"]
            for stage in conv_bias_relu_manifest["program"]["stages"]
        ] == ["convolution_fprop", "binary", "unary"]
        _run_driver(
            driver,
            target,
            conv_bias_relu_request,
            conv_bias_relu_baseline,
            "success",
            "valid_conv_bias_relu_fp16_autotuned",
        )

        conv_bias_relu_tamper_request = (
            fixture_root / "request-conv-bias-relu-alpha-tamper.json"
        )
        conv_bias_relu_tamper_artifact = (
            fixture_root / "conv-bias-relu-alpha-tamper"
        )
        changed_conv_bias_relu = copy.deepcopy(conv_bias_relu_document)
        changed_conv_bias_relu["graph"]["nodes"][1]["attributes"]["alpha"] = 0.5  # type: ignore[index]
        _write_request(conv_bias_relu_tamper_request, changed_conv_bias_relu)
        shutil.copytree(conv_bias_relu_baseline, conv_bias_relu_tamper_artifact)
        changed_conv_bias_relu_manifest = copy.deepcopy(conv_bias_relu_manifest)
        changed_conv_bias_relu_manifest["request_sha256"] = hashlib.sha256(
            conv_bias_relu_tamper_request.read_bytes()
        ).hexdigest()
        (conv_bias_relu_tamper_artifact / "manifest.json").write_text(
            json.dumps(
                changed_conv_bias_relu_manifest, sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            conv_bias_relu_tamper_request,
            conv_bias_relu_tamper_artifact,
            "compilation_failed",
            "conv_bias_relu_coordinated_alpha_tamper",
        )

        def run_convolution_graph_mutation(
            case_name: str,
            mutate: Callable[[dict[str, object]], None],
        ) -> None:
            changed_request_document = copy.deepcopy(convolution_request_document)
            mutate(changed_request_document)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            changed_artifact_path = fixture_root / case_name
            _write_request(changed_request_path, changed_request_document)
            shutil.copytree(convolution_baseline, changed_artifact_path)
            changed_manifest = copy.deepcopy(convolution_manifest)
            changed_manifest["request_sha256"] = hashlib.sha256(
                changed_request_path.read_bytes()
            ).hexdigest()
            (changed_artifact_path / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                changed_artifact_path,
                "compilation_failed",
                case_name,
            )

        def convolution_node(document: dict[str, object]) -> dict[str, Any]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            node = graph["nodes"][0]
            assert isinstance(node, dict)
            return node

        def convolution_tensors(
            document: dict[str, object],
        ) -> list[dict[str, Any]]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            tensors = graph["tensors"]
            assert isinstance(tensors, list)
            return tensors

        def swap_convolution_inputs(document: dict[str, object]) -> None:
            inputs = convolution_node(document)["inputs"]
            assert isinstance(inputs, list)
            inputs[0], inputs[1] = inputs[1], inputs[0]

        def alias_convolution_output(document: dict[str, object]) -> None:
            convolution_node(document)["outputs"][0]["uid"] = 1

        def mismatch_convolution_dtype(document: dict[str, object]) -> None:
            convolution_tensors(document)[1]["data_type"] = "float32"

        def mismatch_convolution_compute(document: dict[str, object]) -> None:
            convolution_node(document)["compute_data_type"] = "float16"

        def mismatch_convolution_rank(document: dict[str, object]) -> None:
            convolution_node(document)["attributes"]["spatial_rank"] = 3

        def mismatch_convolution_groups(document: dict[str, object]) -> None:
            convolution_node(document)["attributes"]["groups"] = 3

        def mismatch_convolution_filter_channels(
            document: dict[str, object],
        ) -> None:
            convolution_tensors(document)[1]["dimensions"][1] = 3

        def mismatch_convolution_output_shape(document: dict[str, object]) -> None:
            convolution_tensors(document)[2]["dimensions"][-1] = 7

        def mismatch_convolution_output_count(document: dict[str, object]) -> None:
            convolution_node(document)["attributes"]["n_outputs"] = 191

        def mismatch_convolution_padding_rank(document: dict[str, object]) -> None:
            convolution_node(document)["attributes"]["pre_padding"] = [1]

        def zero_convolution_stride(document: dict[str, object]) -> None:
            convolution_node(document)["attributes"]["stride"][0] = 0

        def overlap_convolution_input(document: dict[str, object]) -> None:
            convolution_tensors(document)[0]["strides"] = [500, 100, 1, 1]

        def add_convolution_attribute(document: dict[str, object]) -> None:
            convolution_node(document)["attributes"]["mode"] = 0

        for case_name, mutate in (
            ("convolution_graph_role_order_tamper", swap_convolution_inputs),
            ("convolution_graph_output_alias_tamper", alias_convolution_output),
            ("convolution_graph_dtype_tamper", mismatch_convolution_dtype),
            ("convolution_graph_compute_tamper", mismatch_convolution_compute),
            ("convolution_graph_rank_tamper", mismatch_convolution_rank),
            ("convolution_graph_groups_tamper", mismatch_convolution_groups),
            (
                "convolution_graph_filter_channels_tamper",
                mismatch_convolution_filter_channels,
            ),
            (
                "convolution_graph_output_shape_tamper",
                mismatch_convolution_output_shape,
            ),
            (
                "convolution_graph_output_count_tamper",
                mismatch_convolution_output_count,
            ),
            (
                "convolution_graph_padding_rank_tamper",
                mismatch_convolution_padding_rank,
            ),
            ("convolution_graph_zero_stride_tamper", zero_convolution_stride),
            ("convolution_graph_overlap_tamper", overlap_convolution_input),
            ("convolution_graph_extra_attribute_tamper", add_convolution_attribute),
        ):
            run_convolution_graph_mutation(case_name, mutate)

        def run_convolution_manifest_mutation(
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
            *,
            source_request: Path = convolution_request,
            source_artifact: Path = convolution_baseline,
            source_manifest: dict[str, Any] = convolution_manifest,
        ) -> None:
            changed_artifact_path = fixture_root / case_name
            shutil.copytree(source_artifact, changed_artifact_path)
            changed_manifest = copy.deepcopy(source_manifest)
            mutate(changed_manifest)
            (changed_artifact_path / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                source_request,
                changed_artifact_path,
                "compilation_failed",
                case_name,
            )

        def convolution_stage(document: dict[str, Any]) -> dict[str, Any]:
            return document["program"]["stages"][0]

        def convolution_candidate_payload(
            document: dict[str, Any], candidate_index: int = 0
        ) -> dict[str, Any]:
            return convolution_stage(document)["candidates"][candidate_index][
                "payload"
            ]

        convolution_meta_names = (
            [
                "SPATIAL_RANK",
                "GROUPS",
                "INPUT_CHANNELS",
                "OUTPUT_CHANNELS",
                "CHANNELS_PER_GROUP",
            ]
            + [f"INPUT_DIM_{axis}" for axis in range(5)]
            + [f"INPUT_STRIDE_{axis}" for axis in range(5)]
            + [f"FILTER_DIM_{axis}" for axis in range(5)]
            + [f"FILTER_STRIDE_{axis}" for axis in range(5)]
            + [f"OUTPUT_DIM_{axis}" for axis in range(5)]
            + [f"OUTPUT_STRIDE_{axis}" for axis in range(5)]
            + [f"PRE_PADDING_{axis}" for axis in range(3)]
            + [f"POST_PADDING_{axis}" for axis in range(3)]
            + [f"CONV_STRIDE_{axis}" for axis in range(3)]
            + [f"DILATION_{axis}" for axis in range(3)]
            + ["BLOCK_SIZE", "WORKER_COUNT"]
        )

        def set_convolution_meta_and_signature(
            document: dict[str, Any],
            name: str,
            value: int,
            candidate_index: int = 0,
        ) -> None:
            payload = convolution_candidate_payload(document, candidate_index)
            payload["meta"][name] = value
            tokens = payload["full_signature"].split(",")
            tokens[4 + convolution_meta_names.index(name)] = str(value)
            payload["full_signature"] = ",".join(tokens)

        def change_convolution_stage_family(document: dict[str, Any]) -> None:
            convolution_stage(document)["kernel_family"] = "binary"

        def change_convolution_source(document: dict[str, Any]) -> None:
            convolution_stage(document)["kernel"]["source"] = "matmul.py"

        def change_convolution_entry(document: dict[str, Any]) -> None:
            convolution_stage(document)["kernel"]["entry_point"] = (
                "matmul_strided_kernel"
            )

        def change_convolution_abi_name(document: dict[str, Any]) -> None:
            convolution_candidate_payload(document)["argument_abi"][1]["name"] = (
                "weights_ptr"
            )

        def change_convolution_abi_type(document: dict[str, Any]) -> None:
            convolution_candidate_payload(document)["argument_abi"][3]["type"] = (
                "i64"
            )

        def change_convolution_scalar(document: dict[str, Any]) -> None:
            convolution_stage(document)["argument_sources"][3]["value"] = 191

        def change_convolution_groups_meta(document: dict[str, Any]) -> None:
            set_convolution_meta_and_signature(document, "GROUPS", 1)

        def change_convolution_input_dim_meta(document: dict[str, Any]) -> None:
            set_convolution_meta_and_signature(document, "INPUT_DIM_4", 8)

        def change_convolution_filter_stride_meta(document: dict[str, Any]) -> None:
            set_convolution_meta_and_signature(document, "FILTER_STRIDE_3", 7)

        def change_convolution_padding_meta(document: dict[str, Any]) -> None:
            set_convolution_meta_and_signature(document, "PRE_PADDING_1", 0)

        def change_convolution_dilation_meta(document: dict[str, Any]) -> None:
            set_convolution_meta_and_signature(document, "DILATION_2", 1)

        def change_convolution_grid(document: dict[str, Any]) -> None:
            convolution_candidate_payload(document)["grid"][0] += 1

        def change_convolution_fixed_block(document: dict[str, Any]) -> None:
            set_convolution_meta_and_signature(document, "BLOCK_SIZE", 128)

        def duplicate_convolution_candidate(document: dict[str, Any]) -> None:
            candidates = convolution_stage(document)["candidates"]
            candidates.append(copy.deepcopy(candidates[0]))

        def change_convolution_candidate_id(document: dict[str, Any]) -> None:
            convolution_stage(document)["candidates"][0]["candidate_id"] = "other"

        def change_convolution_fixed_selection(document: dict[str, Any]) -> None:
            convolution_stage(document)["autotune"]["selection"][
                "candidate_id"
            ] = "other"

        for case_name, mutate in (
            ("convolution_stage_family_tamper", change_convolution_stage_family),
            ("convolution_stage_source_tamper", change_convolution_source),
            ("convolution_stage_entry_tamper", change_convolution_entry),
            ("convolution_payload_abi_name_tamper", change_convolution_abi_name),
            ("convolution_payload_abi_type_tamper", change_convolution_abi_type),
            ("convolution_scalar_tamper", change_convolution_scalar),
            ("convolution_payload_groups_tamper", change_convolution_groups_meta),
            (
                "convolution_payload_input_dim_tamper",
                change_convolution_input_dim_meta,
            ),
            (
                "convolution_payload_filter_stride_tamper",
                change_convolution_filter_stride_meta,
            ),
            (
                "convolution_payload_padding_tamper",
                change_convolution_padding_meta,
            ),
            (
                "convolution_payload_dilation_tamper",
                change_convolution_dilation_meta,
            ),
            ("convolution_payload_grid_tamper", change_convolution_grid),
            ("convolution_fixed_block_tamper", change_convolution_fixed_block),
            (
                "convolution_fixed_duplicate_candidate_tamper",
                duplicate_convolution_candidate,
            ),
            (
                "convolution_fixed_candidate_id_tamper",
                change_convolution_candidate_id,
            ),
            (
                "convolution_fixed_selection_tamper",
                change_convolution_fixed_selection,
            ),
        ):
            run_convolution_manifest_mutation(case_name, mutate)

        def change_convolution_autotune_table(document: dict[str, Any]) -> None:
            convolution_stage(document)["autotune"]["table"] = "binary"

        def change_convolution_autotune_identity(document: dict[str, Any]) -> None:
            convolution_stage(document)["autotune"]["candidate_identity"] = "0" * 64

        def drop_convolution_autotune_candidate(document: dict[str, Any]) -> None:
            convolution_stage(document)["candidates"].pop()

        for case_name, mutate in (
            (
                "convolution_autotune_table_tamper",
                change_convolution_autotune_table,
            ),
            (
                "convolution_autotune_identity_tamper",
                change_convolution_autotune_identity,
            ),
            (
                "convolution_autotune_candidate_count_tamper",
                drop_convolution_autotune_candidate,
            ),
        ):
            run_convolution_manifest_mutation(
                case_name,
                mutate,
                source_request=convolution_autotune_request,
                source_artifact=convolution_autotune_baseline,
                source_manifest=convolution_autotune_manifest,
            )

        convolution_coordinated_request = (
            fixture_root / "request-convolution-coordinated-tamper.json"
        )
        convolution_coordinated_artifact = (
            fixture_root / "convolution-coordinated-graph-meta-tamper"
        )
        changed_convolution_request = copy.deepcopy(convolution_request_document)
        changed_convolution_request["graph"]["nodes"][0]["attributes"][
            "groups"
        ] = 1
        _write_request(convolution_coordinated_request, changed_convolution_request)
        shutil.copytree(convolution_baseline, convolution_coordinated_artifact)
        changed_convolution_manifest = copy.deepcopy(convolution_manifest)
        changed_convolution_manifest["request_sha256"] = hashlib.sha256(
            convolution_coordinated_request.read_bytes()
        ).hexdigest()
        set_convolution_meta_and_signature(
            changed_convolution_manifest, "GROUPS", 1
        )
        set_convolution_meta_and_signature(
            changed_convolution_manifest, "CHANNELS_PER_GROUP", 4
        )
        (convolution_coordinated_artifact / "manifest.json").write_text(
            json.dumps(changed_convolution_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            convolution_coordinated_request,
            convolution_coordinated_artifact,
            "compilation_failed",
            "convolution_coordinated_graph_meta_tamper",
        )

        convolution_source_tamper = (
            fixture_root / "convolution-coordinated-source-tamper"
        )
        shutil.copytree(convolution_baseline, convolution_source_tamper)
        changed_convolution_source_manifest = copy.deepcopy(convolution_manifest)
        changed_convolution_source_stage = changed_convolution_source_manifest[
            "program"
        ]["stages"][0]
        changed_convolution_kernel = changed_convolution_source_stage["kernel"]
        changed_convolution_source_path = (
            convolution_source_tamper
            / changed_convolution_kernel["materialized_source"]["file"]
        )
        convolution_source_suffix = b"\n# coordinated convolution source tamper\n"
        changed_convolution_source_path.write_bytes(
            changed_convolution_source_path.read_bytes() + convolution_source_suffix
        )
        changed_convolution_source_sha = hashlib.sha256(
            changed_convolution_source_path.read_bytes()
        ).hexdigest()
        changed_convolution_source_name = (
            changed_convolution_source_path.name[:-67]
            + changed_convolution_source_sha
            + ".py"
        )
        changed_convolution_source_path = changed_convolution_source_path.rename(
            changed_convolution_source_path.with_name(
                changed_convolution_source_name
            )
        )
        changed_convolution_kernel["source_sha256"] = changed_convolution_source_sha
        changed_convolution_kernel["materialized_source"][
            "file"
        ] = changed_convolution_source_name
        changed_convolution_kernel["materialized_source"]["size"] += len(
            convolution_source_suffix
        )
        changed_convolution_kernel["materialized_source"][
            "sha256"
        ] = changed_convolution_source_sha
        changed_convolution_source_manifest["source_sha256"] = hashlib.sha256(
            json.dumps(
                [changed_convolution_source_sha], separators=(",", ":")
            ).encode()
        ).hexdigest()
        for candidate in changed_convolution_source_stage["candidates"]:
            candidate["payload"]["source_path"] = changed_convolution_source_name
            candidate["payload"]["source_sha256"] = changed_convolution_source_sha
        (convolution_source_tamper / "manifest.json").write_text(
            json.dumps(
                changed_convolution_source_manifest, sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            convolution_request,
            convolution_source_tamper,
            "compilation_failed",
            "convolution_coordinated_source_tamper",
        )

        matmul_request = fixture_root / "request-matmul-fp16-fixed.json"
        matmul_baseline = fixture_root / "baseline-matmul-fp16-fixed"
        matmul_request_document = _matmul_request(
            identity["identity_sha256"], target
        )
        _write_request(
            matmul_request,
            matmul_request_document,
        )
        provider.compile_request(matmul_request, matmul_baseline, "libtriton_jit")
        matmul_manifest = json.loads(
            (matmul_baseline / "manifest.json").read_bytes()
        )
        matmul_stage = matmul_manifest["program"]["stages"][0]
        matmul_payload = matmul_stage["candidates"][0]["payload"]
        assert matmul_stage["kernel_family"] == "matmul"
        assert matmul_stage["argument_sources"] == [
            {"index": 0, "name": "a_ptr", "source": "binding", "uid": 1,
             "size": 2040, "alignment": 16},
            {"index": 1, "name": "b_ptr", "source": "binding", "uid": 2,
             "size": 2760, "alignment": 16},
            {"index": 2, "name": "output_ptr", "source": "binding", "uid": 3,
             "size": 1564, "alignment": 16},
            {"index": 3, "name": "n_elements", "source": "scalar",
             "type": "i32", "value": 782},
        ]
        assert matmul_payload["entry_point"] == "matmul_strided_kernel"
        assert matmul_payload["meta"]["BATCH"] == 2
        assert matmul_payload["meta"]["M"] == 17
        assert matmul_payload["meta"]["N"] == 23
        assert matmul_payload["meta"]["K"] == 30
        assert matmul_payload["meta"]["BLOCK_SIZE"] == 16
        _run_driver(
            driver,
            target,
            matmul_request,
            matmul_baseline,
            "success",
            "valid_matmul_fp16_fixed",
        )

        matmul_autotune_request = fixture_root / "request-matmul-fp16-autotune.json"
        matmul_autotune_baseline = fixture_root / "baseline-matmul-fp16-autotune"
        _write_request(
            matmul_autotune_request,
            _matmul_request(
                identity["identity_sha256"], target, autotune=True
            ),
        )
        provider.compile_request(
            matmul_autotune_request,
            matmul_autotune_baseline,
            "libtriton_jit",
        )
        matmul_autotune_manifest = json.loads(
            (matmul_autotune_baseline / "manifest.json").read_bytes()
        )
        _run_driver(
            driver,
            target,
            matmul_autotune_request,
            matmul_autotune_baseline,
            "success",
            "valid_matmul_fp16_autotuned",
        )

        matmul_broadcast_request = fixture_root / "request-matmul-bf16-broadcast.json"
        matmul_broadcast_baseline = fixture_root / "baseline-matmul-bf16-broadcast"
        _write_request(
            matmul_broadcast_request,
            _matmul_request(
                identity["identity_sha256"],
                target,
                storage_data_type="bfloat16",
                a_dimensions=[2, 1, 3, 4],
                b_dimensions=[3, 4, 5],
                a_strides=[100, 100, 5, 1],
                b_strides=[100, 6, 1],
                output_strides=[500, 100, 20, 1],
            ),
        )
        provider.compile_request(
            matmul_broadcast_request,
            matmul_broadcast_baseline,
            "libtriton_jit",
        )
        _run_driver(
            driver,
            target,
            matmul_broadcast_request,
            matmul_broadcast_baseline,
            "success",
            "valid_matmul_bf16_broadcast_gapped_fixed",
        )

        matmul_coordinated_request = (
            fixture_root / "request-matmul-coordinated-tamper.json"
        )
        matmul_coordinated_artifact = (
            fixture_root / "matmul-coordinated-graph-meta-tamper"
        )
        changed_matmul_request = copy.deepcopy(matmul_request_document)
        changed_matmul_request["graph"]["nodes"][0]["attributes"]["m"] = 16
        _write_request(matmul_coordinated_request, changed_matmul_request)
        shutil.copytree(matmul_baseline, matmul_coordinated_artifact)
        changed_matmul_manifest = copy.deepcopy(matmul_manifest)
        changed_matmul_manifest["request_sha256"] = hashlib.sha256(
            matmul_coordinated_request.read_bytes()
        ).hexdigest()
        changed_matmul_payload = changed_matmul_manifest["program"]["stages"][0][
            "candidates"
        ][0]["payload"]
        changed_matmul_payload["meta"]["M"] = 16
        changed_matmul_signature = changed_matmul_payload[
            "full_signature"
        ].split(",")
        changed_matmul_signature[5] = "16"
        changed_matmul_payload["full_signature"] = ",".join(
            changed_matmul_signature
        )
        (matmul_coordinated_artifact / "manifest.json").write_text(
            json.dumps(changed_matmul_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            matmul_coordinated_request,
            matmul_coordinated_artifact,
            "compilation_failed",
            "matmul_coordinated_graph_meta_tamper",
        )

        def run_matmul_graph_mutation(
            case_name: str,
            mutate: Callable[[dict[str, object]], None],
        ) -> None:
            changed_request_document = copy.deepcopy(matmul_request_document)
            mutate(changed_request_document)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            changed_artifact_path = fixture_root / case_name
            _write_request(changed_request_path, changed_request_document)
            shutil.copytree(matmul_baseline, changed_artifact_path)
            changed_manifest = copy.deepcopy(matmul_manifest)
            changed_manifest["request_sha256"] = hashlib.sha256(
                changed_request_path.read_bytes()
            ).hexdigest()
            (changed_artifact_path / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                changed_artifact_path,
                "compilation_failed",
                case_name,
            )

        def matmul_node(document: dict[str, object]) -> dict[str, Any]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            node = graph["nodes"][0]
            assert isinstance(node, dict)
            return node

        def matmul_tensors(
            document: dict[str, object],
        ) -> list[dict[str, Any]]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            tensors = graph["tensors"]
            assert isinstance(tensors, list)
            return tensors

        def swap_matmul_inputs(document: dict[str, object]) -> None:
            inputs = matmul_node(document)["inputs"]
            assert isinstance(inputs, list)
            inputs[0], inputs[1] = inputs[1], inputs[0]

        def alias_matmul_output(document: dict[str, object]) -> None:
            outputs = matmul_node(document)["outputs"]
            assert isinstance(outputs, list)
            outputs[0]["uid"] = 1

        def mismatch_matmul_dtype(document: dict[str, object]) -> None:
            matmul_tensors(document)[1]["data_type"] = "float32"

        def mismatch_matmul_k_attribute(document: dict[str, object]) -> None:
            matmul_node(document)["attributes"]["k"] = 29

        def change_matmul_output_shape(document: dict[str, object]) -> None:
            matmul_tensors(document)[2]["dimensions"][-1] = 22

        def change_matmul_a_rank(document: dict[str, object]) -> None:
            tensor = matmul_tensors(document)[0]
            tensor["dimensions"] = [30]
            tensor["strides"] = [1]

        def mismatch_matmul_contraction(document: dict[str, object]) -> None:
            matmul_tensors(document)[1]["dimensions"][-2] = 29

        def add_matmul_attribute(document: dict[str, object]) -> None:
            matmul_node(document)["attributes"]["transpose_a"] = 0

        for case_name, mutate in (
            ("matmul_graph_role_order_tamper", swap_matmul_inputs),
            ("matmul_graph_output_alias_tamper", alias_matmul_output),
            ("matmul_graph_dtype_tamper", mismatch_matmul_dtype),
            ("matmul_graph_k_attribute_tamper", mismatch_matmul_k_attribute),
            ("matmul_graph_output_shape_tamper", change_matmul_output_shape),
            ("matmul_graph_rank_tamper", change_matmul_a_rank),
            ("matmul_graph_contraction_tamper", mismatch_matmul_contraction),
            ("matmul_graph_extra_attribute_tamper", add_matmul_attribute),
        ):
            run_matmul_graph_mutation(case_name, mutate)

        def run_matmul_manifest_mutation(
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
            *,
            source_request: Path = matmul_request,
            source_artifact: Path = matmul_baseline,
            source_manifest: dict[str, Any] = matmul_manifest,
        ) -> None:
            changed_artifact_path = fixture_root / case_name
            shutil.copytree(source_artifact, changed_artifact_path)
            changed_manifest = copy.deepcopy(source_manifest)
            mutate(changed_manifest)
            (changed_artifact_path / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                source_request,
                changed_artifact_path,
                "compilation_failed",
                case_name,
            )

        def matmul_stage(document: dict[str, Any]) -> dict[str, Any]:
            return document["program"]["stages"][0]

        def matmul_payload(
            document: dict[str, Any], candidate_index: int = 0
        ) -> dict[str, Any]:
            return matmul_stage(document)["candidates"][candidate_index]["payload"]

        matmul_meta_names = (
            ["BATCH", "M", "N", "K"]
            + [f"DIM_{axis}" for axis in range(6)]
            + [f"A_BATCH_STRIDE_{axis}" for axis in range(6)]
            + [f"B_BATCH_STRIDE_{axis}" for axis in range(6)]
            + [f"C_BATCH_STRIDE_{axis}" for axis in range(6)]
            + [
                "A_STRIDE_M",
                "A_STRIDE_K",
                "B_STRIDE_K",
                "B_STRIDE_N",
                "C_STRIDE_M",
                "C_STRIDE_N",
                "INPUT_IS_FLOAT32",
                "GROUP_M",
                "BLOCK_SIZE",
                "WORKER_COUNT",
            ]
        )

        def set_matmul_meta_and_signature(
            document: dict[str, Any],
            name: str,
            value: int,
            candidate_index: int = 0,
        ) -> None:
            payload = matmul_payload(document, candidate_index)
            payload["meta"][name] = value
            tokens = payload["full_signature"].split(",")
            tokens[4 + matmul_meta_names.index(name)] = str(value)
            payload["full_signature"] = ",".join(tokens)

        def change_matmul_stage_family(document: dict[str, Any]) -> None:
            matmul_stage(document)["kernel_family"] = "binary"

        def change_matmul_source(document: dict[str, Any]) -> None:
            matmul_stage(document)["kernel"]["source"] = "normalization.py"

        def change_matmul_entry(document: dict[str, Any]) -> None:
            matmul_payload(document)["entry_point"] = "binary_pointwise_kernel"

        def change_matmul_abi_name(document: dict[str, Any]) -> None:
            matmul_payload(document)["argument_abi"][1]["name"] = "rhs_ptr"

        def change_matmul_abi_type(document: dict[str, Any]) -> None:
            matmul_payload(document)["argument_abi"][3]["type"] = "i64"

        def change_matmul_scalar(document: dict[str, Any]) -> None:
            matmul_stage(document)["argument_sources"][3]["value"] = 781

        def change_matmul_m_meta(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "M", 16)

        def change_matmul_dimension_meta(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "DIM_5", 3)

        def change_matmul_batch_stride_meta(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "A_BATCH_STRIDE_5", 0)

        def change_matmul_matrix_stride_meta(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "A_STRIDE_M", 31)

        def change_matmul_input_flag(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "INPUT_IS_FLOAT32", 1)

        def change_matmul_group(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "GROUP_M", 2)

        def change_matmul_grid(document: dict[str, Any]) -> None:
            matmul_payload(document)["grid"][0] += 1

        def change_matmul_fixed_block(document: dict[str, Any]) -> None:
            set_matmul_meta_and_signature(document, "BLOCK_SIZE", 32)

        def duplicate_matmul_candidate(document: dict[str, Any]) -> None:
            candidates = matmul_stage(document)["candidates"]
            candidates.append(copy.deepcopy(candidates[0]))

        def change_matmul_candidate_id(document: dict[str, Any]) -> None:
            matmul_stage(document)["candidates"][0]["candidate_id"] = "alternate"

        def change_matmul_fixed_selection(document: dict[str, Any]) -> None:
            matmul_stage(document)["autotune"]["selection"][
                "candidate_id"
            ] = "alternate"

        for case_name, mutate in (
            ("matmul_stage_family_tamper", change_matmul_stage_family),
            ("matmul_stage_source_tamper", change_matmul_source),
            ("matmul_payload_entry_tamper", change_matmul_entry),
            ("matmul_payload_abi_name_tamper", change_matmul_abi_name),
            ("matmul_payload_abi_type_tamper", change_matmul_abi_type),
            ("matmul_scalar_tamper", change_matmul_scalar),
            ("matmul_payload_m_meta_tamper", change_matmul_m_meta),
            ("matmul_payload_dimension_meta_tamper", change_matmul_dimension_meta),
            (
                "matmul_payload_batch_stride_meta_tamper",
                change_matmul_batch_stride_meta,
            ),
            (
                "matmul_payload_matrix_stride_meta_tamper",
                change_matmul_matrix_stride_meta,
            ),
            ("matmul_payload_input_flag_tamper", change_matmul_input_flag),
            ("matmul_payload_group_tamper", change_matmul_group),
            ("matmul_payload_grid_tamper", change_matmul_grid),
            ("matmul_fixed_block_tamper", change_matmul_fixed_block),
            ("matmul_fixed_duplicate_candidate_tamper", duplicate_matmul_candidate),
            ("matmul_fixed_candidate_id_tamper", change_matmul_candidate_id),
            ("matmul_fixed_selection_tamper", change_matmul_fixed_selection),
        ):
            run_matmul_manifest_mutation(case_name, mutate)

        def change_matmul_autotune_table(document: dict[str, Any]) -> None:
            matmul_stage(document)["autotune"]["table"] = "binary"

        def change_matmul_candidate_identity(document: dict[str, Any]) -> None:
            matmul_stage(document)["autotune"]["candidate_identity"] = "0" * 64

        def drop_matmul_autotune_candidate(document: dict[str, Any]) -> None:
            matmul_stage(document)["candidates"].pop()

        for case_name, mutate in (
            ("matmul_autotune_table_tamper", change_matmul_autotune_table),
            ("matmul_autotune_identity_tamper", change_matmul_candidate_identity),
            ("matmul_autotune_candidate_count_tamper", drop_matmul_autotune_candidate),
        ):
            run_matmul_manifest_mutation(
                case_name,
                mutate,
                source_request=matmul_autotune_request,
                source_artifact=matmul_autotune_baseline,
                source_manifest=matmul_autotune_manifest,
            )

        matmul_source_tamper = fixture_root / "matmul-coordinated-source-tamper"
        shutil.copytree(matmul_baseline, matmul_source_tamper)
        changed_matmul_source_manifest = copy.deepcopy(matmul_manifest)
        changed_matmul_source_stage = changed_matmul_source_manifest["program"][
            "stages"
        ][0]
        changed_matmul_kernel = changed_matmul_source_stage["kernel"]
        changed_matmul_source_path = (
            matmul_source_tamper
            / changed_matmul_kernel["materialized_source"]["file"]
        )
        matmul_source_suffix = b"\n# coordinated MatMul source tamper\n"
        changed_matmul_source_path.write_bytes(
            changed_matmul_source_path.read_bytes() + matmul_source_suffix
        )
        changed_matmul_source_sha = hashlib.sha256(
            changed_matmul_source_path.read_bytes()
        ).hexdigest()
        changed_matmul_source_name = (
            changed_matmul_source_path.name[:-67]
            + changed_matmul_source_sha
            + ".py"
        )
        changed_matmul_source_path.rename(
            changed_matmul_source_path.with_name(changed_matmul_source_name)
        )
        changed_matmul_kernel["source_sha256"] = changed_matmul_source_sha
        changed_matmul_kernel["materialized_source"][
            "file"
        ] = changed_matmul_source_name
        changed_matmul_kernel["materialized_source"]["size"] += len(
            matmul_source_suffix
        )
        changed_matmul_kernel["materialized_source"][
            "sha256"
        ] = changed_matmul_source_sha
        changed_matmul_source_manifest["source_sha256"] = hashlib.sha256(
            json.dumps(
                [changed_matmul_source_sha], separators=(",", ":")
            ).encode()
        ).hexdigest()
        for candidate in changed_matmul_source_stage["candidates"]:
            candidate["payload"]["source_path"] = changed_matmul_source_name
            candidate["payload"]["source_sha256"] = changed_matmul_source_sha
        (matmul_source_tamper / "manifest.json").write_text(
            json.dumps(
                changed_matmul_source_manifest, sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            matmul_request,
            matmul_source_tamper,
            "compilation_failed",
            "matmul_coordinated_source_tamper",
        )

        batchnorm_training_request = fixture_root / "request-batchnorm-training.json"
        batchnorm_training_baseline = fixture_root / "baseline-batchnorm-training"
        _write_request(
            batchnorm_training_request,
            _batchnorm_training_request(identity["identity_sha256"], target),
        )
        provider.compile_request(
            batchnorm_training_request,
            batchnorm_training_baseline,
            "libtriton_jit",
        )
        batchnorm_training_manifest = json.loads(
            (batchnorm_training_baseline / "manifest.json").read_bytes()
        )
        _run_driver(
            driver,
            target,
            batchnorm_training_request,
            batchnorm_training_baseline,
            "success",
            "valid_batchnorm_training_fp16_fixed",
        )
        batchnorm_training_autotune_request = (
            fixture_root / "request-batchnorm-training-autotune.json"
        )
        batchnorm_training_autotune_baseline = (
            fixture_root / "baseline-batchnorm-training-autotune"
        )
        _write_request(
            batchnorm_training_autotune_request,
            _batchnorm_training_request(
                identity["identity_sha256"], target, autotune=True
            ),
        )
        provider.compile_request(
            batchnorm_training_autotune_request,
            batchnorm_training_autotune_baseline,
            "libtriton_jit",
        )
        _run_driver(
            driver,
            target,
            batchnorm_training_autotune_request,
            batchnorm_training_autotune_baseline,
            "success",
            "valid_batchnorm_training_fp16_autotuned",
        )
        batchnorm_training_tamper_request = (
            fixture_root / "request-batchnorm-training-tamper.json"
        )
        batchnorm_training_tamper_artifact = (
            fixture_root / "batchnorm_training_coordinated_tamper"
        )
        changed_training_request = _batchnorm_training_request(
            identity["identity_sha256"], target
        )
        changed_training_request["graph"]["nodes"][0]["attributes"][
            "batch"
        ] = 1
        _write_request(
            batchnorm_training_tamper_request, changed_training_request
        )
        shutil.copytree(
            batchnorm_training_baseline,
            batchnorm_training_tamper_artifact,
        )
        changed_training_manifest = copy.deepcopy(
            batchnorm_training_manifest
        )
        changed_training_manifest["request_sha256"] = hashlib.sha256(
            batchnorm_training_tamper_request.read_bytes()
        ).hexdigest()
        changed_training_payload = changed_training_manifest["program"][
            "stages"
        ][0]["candidates"][0]["payload"]
        changed_training_payload["meta"]["BATCH"] = 1
        changed_training_tokens = changed_training_payload[
            "full_signature"
        ].split(",")
        changed_training_tokens[12] = "1"
        changed_training_payload["full_signature"] = ",".join(
            changed_training_tokens
        )
        (batchnorm_training_tamper_artifact / "manifest.json").write_text(
            json.dumps(changed_training_manifest, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            batchnorm_training_tamper_request,
            batchnorm_training_tamper_artifact,
            "compilation_failed",
            "batchnorm_training_coordinated_tamper",
        )

        rmsnorm_request = fixture_root / "request-rmsnorm.json"
        rmsnorm_baseline = fixture_root / "baseline-rmsnorm"
        rmsnorm_request_document = _rmsnorm_request(
            identity["identity_sha256"], target
        )
        _write_request(rmsnorm_request, rmsnorm_request_document)
        provider.compile_request(
            rmsnorm_request, rmsnorm_baseline, "libtriton_jit"
        )
        rmsnorm_manifest = json.loads(
            (rmsnorm_baseline / "manifest.json").read_bytes()
        )
        rmsnorm_stage = rmsnorm_manifest["program"]["stages"][0]
        rmsnorm_payload = rmsnorm_stage["candidates"][0]["payload"]
        assert rmsnorm_stage["kernel_family"] == "rmsnorm"
        assert rmsnorm_stage["kernel"]["source"] == "normalization.py"
        assert rmsnorm_stage["kernel"]["entry_point"] == (
            "rmsnorm_persistent_kernel"
        )
        assert [
            argument["name"] for argument in rmsnorm_payload["argument_abi"]
        ] == [
            "x_ptr",
            "scale_ptr",
            "bias_ptr",
            "y_ptr",
            "inv_variance_ptr",
            "n_elements",
        ]
        assert rmsnorm_payload["meta"] == {
            "ROWS": 10,
            "NORMALIZED_ELEMENTS": 17,
            "EPSILON": 1.0e-5,
            "BLOCK_SIZE": 256,
            "WORKER_COUNT": aicore_count_from_target(target) * 2,
        }
        _run_driver(
            driver,
            target,
            rmsnorm_request,
            rmsnorm_baseline,
            "success",
            "valid_rmsnorm_fp16_suffix1_fixed",
        )
        rmsnorm_autotune_request = fixture_root / "request-rmsnorm-autotune.json"
        rmsnorm_autotune_baseline = fixture_root / "baseline-rmsnorm-autotune"
        rmsnorm_autotune_document = copy.deepcopy(rmsnorm_request_document)
        rmsnorm_autotune_document["build_options"]["autotune"] = True
        _write_request(rmsnorm_autotune_request, rmsnorm_autotune_document)
        provider.compile_request(
            rmsnorm_autotune_request,
            rmsnorm_autotune_baseline,
            "libtriton_jit",
        )
        rmsnorm_autotune_manifest = json.loads(
            (rmsnorm_autotune_baseline / "manifest.json").read_bytes()
        )
        rmsnorm_autotune_stage = rmsnorm_autotune_manifest["program"][
            "stages"
        ][0]
        assert rmsnorm_autotune_stage["autotune"]["table"] == "rmsnorm"
        assert [
            candidate["payload"]["meta"]["BLOCK_SIZE"]
            for candidate in rmsnorm_autotune_stage["candidates"]
        ] == [256, 128]
        _run_driver(
            driver,
            target,
            rmsnorm_autotune_request,
            rmsnorm_autotune_baseline,
            "success",
            "valid_rmsnorm_fp16_suffix1_autotuned",
        )

        layernorm_request = fixture_root / "request-layernorm.json"
        layernorm_baseline = fixture_root / "baseline-layernorm"
        layernorm_request_document = _layernorm_request(
            identity["identity_sha256"], target
        )
        _write_request(layernorm_request, layernorm_request_document)
        provider.compile_request(
            layernorm_request, layernorm_baseline, "libtriton_jit"
        )
        layernorm_manifest = json.loads(
            (layernorm_baseline / "manifest.json").read_bytes()
        )
        layernorm_stage = layernorm_manifest["program"]["stages"][0]
        layernorm_payload = layernorm_stage["candidates"][0]["payload"]
        assert layernorm_stage["kernel_family"] == "layernorm"
        assert layernorm_stage["kernel"]["source"] == "normalization.py"
        assert layernorm_stage["kernel"]["entry_point"] == (
            "layernorm_persistent_kernel"
        )
        assert [
            argument["name"]
            for argument in layernorm_payload["argument_abi"]
        ] == [
            "x_ptr",
            "scale_ptr",
            "bias_ptr",
            "y_ptr",
            "mean_ptr",
            "inv_variance_ptr",
            "n_elements",
        ]
        assert layernorm_payload["meta"] == {
            "ROWS": 10,
            "NORMALIZED_ELEMENTS": 17,
            "EPSILON": 1.0e-5,
            "BLOCK_SIZE": 256,
            "WORKER_COUNT": aicore_count_from_target(target) * 2,
        }
        _run_driver(
            driver,
            target,
            layernorm_request,
            layernorm_baseline,
            "success",
            "valid_layernorm_fp16_suffix1_fixed",
        )

        layernorm_variants = [
            (
                "valid_layernorm_fp32_suffix2_fixed",
                {"data_type": "float32", "dimensions": [3, 4, 5],
                 "normalized_rank": 2},
                3,
                20,
                False,
                0,
            ),
            (
                "valid_layernorm_bf16_rank8_suffix3_autotuned",
                {"data_type": "bfloat16",
                 "dimensions": [1, 2, 1, 2, 1, 2, 1, 3],
                 "normalized_rank": 3, "autotune": True},
                4,
                6,
                True,
                0,
            ),
            (
                "valid_layernorm_fp16_virtual_y_fixed",
                {"virtual_y": True},
                10,
                17,
                False,
                512,
            ),
        ]
        for (
            case_name,
            options,
            expected_rows,
            expected_normalized,
            expected_autotune,
            expected_workspace,
        ) in layernorm_variants:
            variant_request = fixture_root / f"request-{case_name}.json"
            variant_artifact = fixture_root / case_name
            variant_document = _layernorm_request(
                identity["identity_sha256"], target, **options
            )
            _write_request(variant_request, variant_document)
            provider.compile_request(
                variant_request, variant_artifact, "libtriton_jit"
            )
            variant_manifest = json.loads(
                (variant_artifact / "manifest.json").read_bytes()
            )
            variant_stage = variant_manifest["program"]["stages"][0]
            assert variant_manifest["workspace_size"] == expected_workspace
            assert variant_stage["kernel_family"] == "layernorm"
            assert variant_stage["autotune"]["enabled"] is expected_autotune
            assert [
                candidate["payload"]["meta"]["BLOCK_SIZE"]
                for candidate in variant_stage["candidates"]
            ] == ([256, 128] if expected_autotune else [256])
            for candidate in variant_stage["candidates"]:
                assert candidate["payload"]["meta"]["ROWS"] == expected_rows
                assert candidate["payload"]["meta"][
                    "NORMALIZED_ELEMENTS"
                ] == expected_normalized
            _run_driver(
                driver,
                target,
                variant_request,
                variant_artifact,
                "success",
                case_name,
            )

        layernorm_coordinated_request = (
            fixture_root / "request-layernorm-coordinated-tamper.json"
        )
        layernorm_coordinated_artifact = (
            fixture_root / "layernorm_coordinated_request_tamper"
        )
        changed_layernorm_request = copy.deepcopy(layernorm_request_document)
        changed_layernorm_request["graph"]["nodes"][0]["attributes"][
            "rows"
        ] = 9
        _write_request(layernorm_coordinated_request, changed_layernorm_request)
        shutil.copytree(layernorm_baseline, layernorm_coordinated_artifact)
        changed_layernorm_manifest = copy.deepcopy(layernorm_manifest)
        changed_layernorm_manifest["request_sha256"] = hashlib.sha256(
            layernorm_coordinated_request.read_bytes()
        ).hexdigest()
        changed_layernorm_stage = changed_layernorm_manifest["program"][
            "stages"
        ][0]
        for candidate in changed_layernorm_stage["candidates"]:
            changed_payload = candidate["payload"]
            changed_payload["meta"]["ROWS"] = 9
            changed_tokens = changed_payload["full_signature"].split(",")
            changed_tokens[7] = "9"
            changed_payload["full_signature"] = ",".join(changed_tokens)
        (layernorm_coordinated_artifact / "manifest.json").write_text(
            json.dumps(changed_layernorm_manifest, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            layernorm_coordinated_request,
            layernorm_coordinated_artifact,
            "compilation_failed",
            "layernorm_coordinated_request_tamper",
        )

        def run_layernorm_graph_mutation(
            case_name: str,
            mutate: Callable[[dict[str, object]], None],
        ) -> None:
            changed_document = copy.deepcopy(layernorm_request_document)
            mutate(changed_document)
            changed_request = fixture_root / f"request-{case_name}.json"
            changed_artifact = fixture_root / case_name
            _write_request(changed_request, changed_document)
            shutil.copytree(layernorm_baseline, changed_artifact)
            changed_manifest = copy.deepcopy(layernorm_manifest)
            changed_manifest["request_sha256"] = hashlib.sha256(
                changed_request.read_bytes()
            ).hexdigest()
            (changed_artifact / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request,
                changed_artifact,
                "compilation_failed",
                case_name,
            )

        def layernorm_graph(document: dict[str, object]) -> dict[str, Any]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            return graph

        run_layernorm_graph_mutation(
            "layernorm_graph_port_order_tamper",
            lambda document: layernorm_graph(document)["nodes"][0]["inputs"].__setitem__(
                slice(1, 3),
                list(
                    reversed(
                        layernorm_graph(document)["nodes"][0]["inputs"][1:3]
                    )
                ),
            ),
        )
        run_layernorm_graph_mutation(
            "layernorm_graph_duplicate_uid_tamper",
            lambda document: layernorm_graph(document)["nodes"][0]["outputs"][1].__setitem__(
                "uid", 4
            ),
        )
        run_layernorm_graph_mutation(
            "layernorm_graph_statistic_dtype_tamper",
            lambda document: layernorm_graph(document)["tensors"][4].__setitem__(
                "data_type", "float16"
            ),
        )
        run_layernorm_graph_mutation(
            "layernorm_graph_normalized_extent_tamper",
            lambda document: layernorm_graph(document)["nodes"][0][
                "attributes"
            ].__setitem__("normalized_elements", 16),
        )
        run_layernorm_graph_mutation(
            "layernorm_graph_parameter_shape_tamper",
            lambda document: (
                layernorm_graph(document)["tensors"][1].__setitem__(
                    "dimensions", [1, 17, 1]
                ),
                layernorm_graph(document)["tensors"][1].__setitem__(
                    "strides", [17, 1, 1]
                ),
            ),
        )

        batchnorm_request = fixture_root / "request-batchnorm-inference.json"
        batchnorm_baseline = fixture_root / "baseline-batchnorm-inference"
        batchnorm_request_document = _batchnorm_inference_request(
            identity["identity_sha256"], target
        )
        _write_request(
            batchnorm_request,
            batchnorm_request_document,
        )
        provider.compile_request(
            batchnorm_request, batchnorm_baseline, "libtriton_jit"
        )
        batchnorm_manifest = json.loads(
            (batchnorm_baseline / "manifest.json").read_bytes()
        )
        batchnorm_stage = batchnorm_manifest["program"]["stages"][0]
        batchnorm_payload = batchnorm_stage["candidates"][0]["payload"]
        assert batchnorm_stage["kernel_family"] == "batchnorm_inference"
        assert batchnorm_stage["kernel"]["source"] == "normalization.py"
        assert batchnorm_stage["kernel"]["entry_point"] == (
            "batchnorm_inference_nchw_persistent_kernel"
        )
        assert batchnorm_payload["argument_abi"] == [
            {"index": 0, "name": "x_ptr", "type": "pointer"},
            {"index": 1, "name": "mean_ptr", "type": "pointer"},
            {"index": 2, "name": "inv_variance_ptr", "type": "pointer"},
            {"index": 3, "name": "scale_ptr", "type": "pointer"},
            {"index": 4, "name": "bias_ptr", "type": "pointer"},
            {"index": 5, "name": "y_ptr", "type": "pointer"},
            {"index": 6, "name": "n_elements", "type": "i32"},
        ]
        assert batchnorm_payload["meta"] == {
            "RANK": 2,
            "CHANNELS": 3,
            "SPATIAL": 1,
            "BLOCK_SIZE": 256,
            "WORKER_COUNT": aicore_count_from_target(target) * 2,
        }
        assert batchnorm_stage["autotune"] == {
            "schema_version": 1,
            "enabled": False,
            "selection": {"state": "fixed", "candidate_id": "default"},
        }
        _run_driver(
            driver,
            target,
            batchnorm_request,
            batchnorm_baseline,
            "success",
            "valid_batchnorm_inference_fp16_fast_fixed",
        )

        batchnorm_requested_autotune = (
            fixture_root / "request-batchnorm-requested-autotune.json"
        )
        batchnorm_requested_autotune_artifact = (
            fixture_root / "baseline-batchnorm-requested-autotune"
        )
        requested_autotune_document = copy.deepcopy(
            batchnorm_request_document
        )
        requested_autotune_document["build_options"]["autotune"] = True
        _write_request(
            batchnorm_requested_autotune, requested_autotune_document
        )
        provider.compile_request(
            batchnorm_requested_autotune,
            batchnorm_requested_autotune_artifact,
            "libtriton_jit",
        )
        requested_autotune_manifest = json.loads(
            (batchnorm_requested_autotune_artifact / "manifest.json").read_bytes()
        )
        requested_autotune_stage = requested_autotune_manifest["program"][
            "stages"
        ][0]
        requested_autotune_policy = requested_autotune_stage["autotune"]
        assert requested_autotune_policy["enabled"] is True
        assert requested_autotune_policy["selection"] == {
            "state": "pending",
            "candidate_id": "",
        }
        assert requested_autotune_policy["table"] == "batchnorm_inference"
        assert len(requested_autotune_policy["candidate_identity"]) == 64
        assert [
            candidate["payload"]["meta"]["BLOCK_SIZE"]
            for candidate in requested_autotune_stage["candidates"]
        ] == [256, 128]
        assert len(
            {
                candidate["candidate_id"]
                for candidate in requested_autotune_stage["candidates"]
            }
        ) == 2
        _run_driver(
            driver,
            target,
            batchnorm_requested_autotune,
            batchnorm_requested_autotune_artifact,
            "success",
            "valid_batchnorm_inference_fp16_fast_autotuned",
        )
        reordered_autotune_artifact = (
            fixture_root / "valid-batchnorm-autotune-reordered"
        )
        shutil.copytree(
            batchnorm_requested_autotune_artifact,
            reordered_autotune_artifact,
        )
        reordered_autotune_manifest = copy.deepcopy(
            requested_autotune_manifest
        )
        reordered_autotune_manifest["program"]["stages"][0][
            "candidates"
        ].reverse()
        (reordered_autotune_artifact / "manifest.json").write_text(
            json.dumps(reordered_autotune_manifest, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            batchnorm_requested_autotune,
            reordered_autotune_artifact,
            "success",
            "valid_batchnorm_inference_fp16_fast_autotuned",
        )

        for case_name, parameter_dimensions, parameter_strides in (
            ("leading_one", [1, 3], [3, 1]),
            ("trailing_one", [3, 1], [1, 1]),
            ("common_nchw", [1, 3, 1, 1], [3, 1, 1, 1]),
        ):
            shaped_document = copy.deepcopy(batchnorm_request_document)
            shaped_graph = shaped_document["graph"]
            assert isinstance(shaped_graph, dict)
            shaped_tensors = shaped_graph["tensors"]
            assert isinstance(shaped_tensors, list)
            for tensor_index in range(1, 5):
                shaped_tensors[tensor_index]["dimensions"] = parameter_dimensions
                shaped_tensors[tensor_index]["strides"] = parameter_strides
            shaped_request = fixture_root / f"request-batchnorm-{case_name}.json"
            shaped_artifact = fixture_root / f"baseline-batchnorm-{case_name}"
            _write_request(shaped_request, shaped_document)
            provider.compile_request(
                shaped_request, shaped_artifact, "libtriton_jit"
            )
            _run_driver(
                driver,
                target,
                shaped_request,
                shaped_artifact,
                "success",
                f"valid_batchnorm_parameter_shape_{case_name}",
            )

        for case, request_document, expected_rank, expected_spatial in (
            (
                "fp32_generic_fixed",
                _batchnorm_inference_request(
                    identity["identity_sha256"],
                    target,
                    storage_data_type="float32",
                    dimensions=[2, 3, 2, 2, 2],
                    x_strides=[100, 30, 12, 5, 1],
                    y_strides=[110, 32, 13, 5, 1],
                ),
                5,
                8,
            ),
            (
                "bf16_generic_fixed",
                _batchnorm_inference_request(
                    identity["identity_sha256"],
                    target,
                    storage_data_type="bfloat16",
                    dimensions=[2, 3, 2],
                    x_strides=[30, 8, 2],
                    y_strides=[40, 10, 3],
                ),
                3,
                2,
            ),
        ):
            generic_request = fixture_root / f"request-batchnorm-{case}.json"
            generic_baseline = fixture_root / f"baseline-batchnorm-{case}"
            _write_request(generic_request, request_document)
            provider.compile_request(
                generic_request, generic_baseline, "libtriton_jit"
            )
            generic_manifest = json.loads(
                (generic_baseline / "manifest.json").read_bytes()
            )
            generic_stage = generic_manifest["program"]["stages"][0]
            generic_payload = generic_stage["candidates"][0]["payload"]
            assert generic_stage["kernel_family"] == "batchnorm_inference"
            assert generic_stage["kernel"]["entry_point"] == (
                "batchnorm_inference_strided_persistent_kernel"
            )
            assert generic_payload["meta"]["RANK"] == expected_rank
            assert generic_payload["meta"]["CHANNELS"] == 3
            assert generic_payload["meta"]["SPATIAL"] == expected_spatial
            assert generic_payload["meta"]["BLOCK_SIZE"] == 256
            _run_driver(
                driver,
                target,
                generic_request,
                generic_baseline,
                "success",
                f"valid_batchnorm_inference_{case}",
            )

        virtual_batchnorm_request = (
            fixture_root / "request-batchnorm-virtual-dag.json"
        )
        virtual_batchnorm_baseline = (
            fixture_root / "baseline-batchnorm-virtual-dag"
        )
        _write_request(
            virtual_batchnorm_request,
            _batchnorm_virtual_dag_request(
                identity["identity_sha256"], target
            ),
        )
        provider.compile_request(
            virtual_batchnorm_request,
            virtual_batchnorm_baseline,
            "libtriton_jit",
        )
        virtual_batchnorm_manifest = json.loads(
            (virtual_batchnorm_baseline / "manifest.json").read_bytes()
        )
        _run_driver(
            driver,
            target,
            virtual_batchnorm_request,
            virtual_batchnorm_baseline,
            "success",
            "valid_batchnorm_inference_virtual_dag_fixed",
        )

        batchnorm_source_tamper = (
            fixture_root / "batchnorm_fully_coordinated_source_tamper"
        )
        shutil.copytree(batchnorm_baseline, batchnorm_source_tamper)
        changed_source_manifest = copy.deepcopy(batchnorm_manifest)
        changed_source_stage = changed_source_manifest["program"]["stages"][0]
        changed_source_kernel = changed_source_stage["kernel"]
        changed_source_path = (
            batchnorm_source_tamper
            / changed_source_kernel["materialized_source"]["file"]
        )
        suffix = b"\n# fully coordinated source tamper\n"
        changed_source_path.write_bytes(changed_source_path.read_bytes() + suffix)
        changed_source_sha = hashlib.sha256(
            changed_source_path.read_bytes()
        ).hexdigest()
        changed_source_name = (
            changed_source_path.name[:-67] + changed_source_sha + ".py"
        )
        changed_source_path.rename(changed_source_path.with_name(changed_source_name))
        changed_source_kernel["source_sha256"] = changed_source_sha
        changed_source_kernel["materialized_source"]["file"] = changed_source_name
        changed_source_kernel["materialized_source"]["size"] += len(suffix)
        changed_source_kernel["materialized_source"]["sha256"] = changed_source_sha
        changed_source_manifest["source_sha256"] = hashlib.sha256(
            json.dumps([changed_source_sha], separators=(",", ":")).encode()
        ).hexdigest()
        for candidate in changed_source_stage["candidates"]:
            candidate["payload"]["source_path"] = changed_source_name
            candidate["payload"]["source_sha256"] = changed_source_sha
        (batchnorm_source_tamper / "manifest.json").write_text(
            json.dumps(changed_source_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            batchnorm_request,
            batchnorm_source_tamper,
            "compilation_failed",
            "batchnorm_fully_coordinated_source_tamper",
        )

        batchnorm_coordinated_tamper = (
            fixture_root / "batchnorm_coordinated_request_tamper"
        )
        shutil.copytree(batchnorm_baseline, batchnorm_coordinated_tamper)
        changed_batchnorm = copy.deepcopy(batchnorm_manifest)
        changed_stage = changed_batchnorm["program"]["stages"][0]
        changed_stage["argument_sources"][0]["size"] = 14
        changed_payload = changed_stage["candidates"][0]["payload"]
        changed_payload["meta"]["CHANNELS"] = 2
        changed_tokens = changed_payload["full_signature"].split(",")
        changed_tokens[8] = "2"
        changed_payload["full_signature"] = ",".join(changed_tokens)
        (batchnorm_coordinated_tamper / "manifest.json").write_text(
            json.dumps(changed_batchnorm, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            batchnorm_request,
            batchnorm_coordinated_tamper,
            "compilation_failed",
            "batchnorm_coordinated_request_tamper",
        )

        def run_batchnorm_graph_mutation(
            case_name: str,
            mutate: Callable[[dict[str, object]], None],
        ) -> None:
            changed_request_document = copy.deepcopy(
                batchnorm_request_document
            )
            mutate(changed_request_document)
            changed_request = fixture_root / f"request-{case_name}.json"
            changed_artifact = fixture_root / case_name
            _write_request(changed_request, changed_request_document)
            shutil.copytree(batchnorm_baseline, changed_artifact)
            changed_manifest = copy.deepcopy(batchnorm_manifest)
            changed_manifest["request_sha256"] = hashlib.sha256(
                changed_request.read_bytes()
            ).hexdigest()
            (changed_artifact / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request,
                changed_artifact,
                "compilation_failed",
                case_name,
            )

        def batchnorm_node(document: dict[str, object]) -> dict[str, Any]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            node = graph["nodes"][0]
            assert isinstance(node, dict)
            return node

        def batchnorm_tensors(
            document: dict[str, object],
        ) -> list[dict[str, Any]]:
            graph = document["graph"]
            assert isinstance(graph, dict)
            tensors = graph["tensors"]
            assert isinstance(tensors, list)
            return tensors

        def swap_batchnorm_roles(document: dict[str, object]) -> None:
            inputs = batchnorm_node(document)["inputs"]
            assert isinstance(inputs, list)
            inputs[0], inputs[1] = inputs[1], inputs[0]

        def change_batchnorm_output_role(document: dict[str, object]) -> None:
            outputs = batchnorm_node(document)["outputs"]
            assert isinstance(outputs, list)
            outputs[0]["name"] = "output"

        def drop_batchnorm_spatial(document: dict[str, object]) -> None:
            del batchnorm_node(document)["attributes"]["spatial"]

        def add_batchnorm_attribute(document: dict[str, object]) -> None:
            batchnorm_node(document)["attributes"]["epsilon"] = 1.0e-5

        def mismatch_batchnorm_channels(document: dict[str, object]) -> None:
            batchnorm_node(document)["attributes"]["channels"] = 2

        def change_batchnorm_parameter_dtype(
            document: dict[str, object],
        ) -> None:
            batchnorm_tensors(document)[1]["data_type"] = "float16"

        def mismatch_batchnorm_xy_dtype(document: dict[str, object]) -> None:
            batchnorm_tensors(document)[5]["data_type"] = "float32"

        def change_batchnorm_parameter_shape(
            document: dict[str, object],
        ) -> None:
            parameter = batchnorm_tensors(document)[1]
            parameter["dimensions"] = [1, 4]
            parameter["strides"] = [4, 1]

        def change_batchnorm_parameter_stride(
            document: dict[str, object],
        ) -> None:
            batchnorm_tensors(document)[1]["strides"] = [2]

        def change_batchnorm_x_stride(document: dict[str, object]) -> None:
            batchnorm_tensors(document)[0]["strides"] = [4, 1]
            batchnorm_node(document)["attributes"]["x_strides"] = [4, 1]

        def change_batchnorm_y_stride(document: dict[str, object]) -> None:
            batchnorm_tensors(document)[5]["strides"] = [4, 1]
            batchnorm_node(document)["attributes"]["y_strides"] = [4, 1]

        for case_name, mutate in (
            ("batchnorm_graph_role_order_tamper", swap_batchnorm_roles),
            ("batchnorm_graph_y_role_tamper", change_batchnorm_output_role),
            ("batchnorm_graph_missing_attr_tamper", drop_batchnorm_spatial),
            ("batchnorm_graph_extra_attr_tamper", add_batchnorm_attribute),
            ("batchnorm_graph_channels_tamper", mismatch_batchnorm_channels),
            (
                "batchnorm_graph_parameter_dtype_tamper",
                change_batchnorm_parameter_dtype,
            ),
            ("batchnorm_graph_xy_dtype_tamper", mismatch_batchnorm_xy_dtype),
            (
                "batchnorm_graph_parameter_shape_tamper",
                change_batchnorm_parameter_shape,
            ),
            (
                "batchnorm_graph_parameter_stride_tamper",
                change_batchnorm_parameter_stride,
            ),
            ("batchnorm_graph_x_stride_tamper", change_batchnorm_x_stride),
            ("batchnorm_graph_y_stride_tamper", change_batchnorm_y_stride),
        ):
            run_batchnorm_graph_mutation(case_name, mutate)

        def run_batchnorm_manifest_mutation(
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
            *,
            source_request: Path = batchnorm_request,
            source_artifact: Path = batchnorm_baseline,
            source_manifest: dict[str, Any] = batchnorm_manifest,
        ) -> None:
            changed_artifact = fixture_root / case_name
            shutil.copytree(source_artifact, changed_artifact)
            changed_manifest = copy.deepcopy(source_manifest)
            mutate(changed_manifest)
            (changed_artifact / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                source_request,
                changed_artifact,
                "compilation_failed",
                case_name,
            )

        def batchnorm_payload(document: dict[str, Any]) -> dict[str, Any]:
            return document["program"]["stages"][0]["candidates"][0][
                "payload"
            ]

        def set_batchnorm_meta_and_signature(
            document: dict[str, Any],
            name: str,
            value: int,
            token_index: int,
        ) -> None:
            payload = batchnorm_payload(document)
            payload["meta"][name] = value
            tokens = payload["full_signature"].split(",")
            tokens[token_index] = str(value)
            payload["full_signature"] = ",".join(tokens)

        def drop_batchnorm_abi_slot(document: dict[str, Any]) -> None:
            batchnorm_payload(document)["argument_abi"].pop()

        def change_batchnorm_abi_name(document: dict[str, Any]) -> None:
            batchnorm_payload(document)["argument_abi"][2]["name"] = (
                "variance_ptr"
            )

        def change_batchnorm_abi_type(document: dict[str, Any]) -> None:
            batchnorm_payload(document)["argument_abi"][4]["type"] = "i32"

        def change_batchnorm_pointer_signature(
            document: dict[str, Any],
        ) -> None:
            payload = batchnorm_payload(document)
            tokens = payload["full_signature"].split(",")
            tokens[1] = "*fp16:16"
            payload["full_signature"] = ",".join(tokens)

        def change_batchnorm_rank_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "RANK", 3, 7)

        def change_batchnorm_channels_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "CHANNELS", 4, 8)

        def change_batchnorm_spatial_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "SPATIAL", 2, 9)

        def change_batchnorm_block_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "BLOCK_SIZE", 128, 10)

        def change_batchnorm_worker_meta(document: dict[str, Any]) -> None:
            payload = batchnorm_payload(document)
            set_batchnorm_meta_and_signature(
                document,
                "WORKER_COUNT",
                int(payload["meta"]["WORKER_COUNT"]) + 1,
                11,
            )

        def change_batchnorm_grid(document: dict[str, Any]) -> None:
            batchnorm_payload(document)["grid"][0] = 5

        def change_batchnorm_entry(document: dict[str, Any]) -> None:
            batchnorm_payload(document)["entry_point"] = (
                "batchnorm_inference_strided_persistent_kernel"
            )

        for case_name, mutate in (
            ("batchnorm_payload_abi_count_tamper", drop_batchnorm_abi_slot),
            ("batchnorm_payload_abi_name_tamper", change_batchnorm_abi_name),
            ("batchnorm_payload_abi_type_tamper", change_batchnorm_abi_type),
            (
                "batchnorm_payload_pointer_signature_tamper",
                change_batchnorm_pointer_signature,
            ),
            ("batchnorm_payload_rank_meta_tamper", change_batchnorm_rank_meta),
            (
                "batchnorm_payload_channels_meta_tamper",
                change_batchnorm_channels_meta,
            ),
            (
                "batchnorm_payload_spatial_meta_tamper",
                change_batchnorm_spatial_meta,
            ),
            (
                "batchnorm_payload_fixed_block_tamper",
                change_batchnorm_block_meta,
            ),
            (
                "batchnorm_payload_worker_meta_tamper",
                change_batchnorm_worker_meta,
            ),
            ("batchnorm_payload_grid_tamper", change_batchnorm_grid),
            ("batchnorm_payload_entry_tamper", change_batchnorm_entry),
        ):
            run_batchnorm_manifest_mutation(case_name, mutate)

        def change_batchnorm_dimension_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "DIM_7", 3, 17)

        def change_batchnorm_x_stride_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "X_STRIDE_7", 4, 25)

        def change_batchnorm_y_stride_meta(document: dict[str, Any]) -> None:
            set_batchnorm_meta_and_signature(document, "Y_STRIDE_7", 4, 33)

        for case_name, mutate in (
            (
                "batchnorm_payload_dimension_meta_tamper",
                change_batchnorm_dimension_meta,
            ),
            (
                "batchnorm_payload_x_stride_meta_tamper",
                change_batchnorm_x_stride_meta,
            ),
            (
                "batchnorm_payload_y_stride_meta_tamper",
                change_batchnorm_y_stride_meta,
            ),
        ):
            run_batchnorm_manifest_mutation(
                case_name,
                mutate,
                source_request=generic_request,
                source_artifact=generic_baseline,
                source_manifest=generic_manifest,
            )

        def virtual_batchnorm_stage(
            document: dict[str, Any],
        ) -> dict[str, Any]:
            return document["program"]["stages"][1]

        def change_batchnorm_graph_workspace_size(
            document: dict[str, Any],
        ) -> None:
            document["workspace_size"] = 256

        def change_batchnorm_stage_workspace(
            document: dict[str, Any],
        ) -> None:
            virtual_batchnorm_stage(document)["workspace"]["size"] += 1

        def change_batchnorm_dependency(document: dict[str, Any]) -> None:
            virtual_batchnorm_stage(document)["dependencies"] = []

        def change_batchnorm_x_workspace_offset(
            document: dict[str, Any],
        ) -> None:
            virtual_batchnorm_stage(document)["argument_sources"][0][
                "offset"
            ] = 256

        def change_batchnorm_y_workspace_offset(
            document: dict[str, Any],
        ) -> None:
            virtual_batchnorm_stage(document)["argument_sources"][5][
                "offset"
            ] = 0

        def change_batchnorm_downstream_offset(
            document: dict[str, Any],
        ) -> None:
            document["program"]["stages"][2]["argument_sources"][0][
                "offset"
            ] = 0

        def change_batchnorm_x_source_kind(
            document: dict[str, Any],
        ) -> None:
            source = virtual_batchnorm_stage(document)["argument_sources"][0]
            source["source"] = "binding"
            del source["offset"]

        def change_batchnorm_x_source_uid(document: dict[str, Any]) -> None:
            virtual_batchnorm_stage(document)["argument_sources"][0]["uid"] = 7

        for case_name, mutate in (
            (
                "batchnorm_virtual_graph_workspace_size_tamper",
                change_batchnorm_graph_workspace_size,
            ),
            (
                "batchnorm_virtual_stage_workspace_tamper",
                change_batchnorm_stage_workspace,
            ),
            ("batchnorm_virtual_dependency_tamper", change_batchnorm_dependency),
            (
                "batchnorm_virtual_x_workspace_offset_tamper",
                change_batchnorm_x_workspace_offset,
            ),
            (
                "batchnorm_virtual_y_workspace_offset_tamper",
                change_batchnorm_y_workspace_offset,
            ),
            (
                "batchnorm_virtual_downstream_offset_tamper",
                change_batchnorm_downstream_offset,
            ),
            (
                "batchnorm_virtual_x_source_kind_tamper",
                change_batchnorm_x_source_kind,
            ),
            (
                "batchnorm_virtual_x_source_uid_tamper",
                change_batchnorm_x_source_uid,
            ),
        ):
            run_batchnorm_manifest_mutation(
                case_name,
                mutate,
                source_request=virtual_batchnorm_request,
                source_artifact=virtual_batchnorm_baseline,
                source_manifest=virtual_batchnorm_manifest,
            )

        def batchnorm_stage(document: dict[str, Any]) -> dict[str, Any]:
            return document["program"]["stages"][0]

        def drop_batchnorm_candidate(document: dict[str, Any]) -> None:
            batchnorm_stage(document)["candidates"] = []

        def duplicate_batchnorm_candidate(document: dict[str, Any]) -> None:
            candidates = batchnorm_stage(document)["candidates"]
            candidates.append(copy.deepcopy(candidates[0]))

        def change_batchnorm_candidate_id(document: dict[str, Any]) -> None:
            batchnorm_stage(document)["candidates"][0]["candidate_id"] = (
                "alternate"
            )

        def change_batchnorm_fixed_selection(document: dict[str, Any]) -> None:
            batchnorm_stage(document)["autotune"]["selection"][
                "candidate_id"
            ] = "alternate"

        def enable_batchnorm_autotune(document: dict[str, Any]) -> None:
            autotune = batchnorm_stage(document)["autotune"]
            autotune["enabled"] = True
            autotune["selection"] = {"state": "pending", "candidate_id": ""}

        def change_batchnorm_num_warps(document: dict[str, Any]) -> None:
            batchnorm_payload(document)["compile_options"]["num_warps"] = 8

        def change_batchnorm_stage_family(document: dict[str, Any]) -> None:
            batchnorm_stage(document)["kernel_family"] = "binary"

        for case_name, mutate in (
            ("batchnorm_fixed_empty_candidate_tamper", drop_batchnorm_candidate),
            (
                "batchnorm_fixed_duplicate_candidate_tamper",
                duplicate_batchnorm_candidate,
            ),
            ("batchnorm_fixed_candidate_id_tamper", change_batchnorm_candidate_id),
            (
                "batchnorm_fixed_selection_tamper",
                change_batchnorm_fixed_selection,
            ),
            ("batchnorm_fixed_autotune_tamper", enable_batchnorm_autotune),
            ("batchnorm_fixed_num_warps_tamper", change_batchnorm_num_warps),
            ("batchnorm_stage_family_tamper", change_batchnorm_stage_family),
        ):
            run_batchnorm_manifest_mutation(case_name, mutate)

        reduction_fixtures: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
        for operation in REDUCTION_MODES:
            for storage_data_type in ("float32", "float16", "bfloat16"):
                case = f"{operation}_{storage_data_type}"
                reduction_request = fixture_root / f"request-{case}.json"
                reduction_baseline = fixture_root / f"baseline-{case}"
                _write_request(
                    reduction_request,
                    _reduction_request(
                        identity["identity_sha256"],
                        target,
                        operation=operation,
                        storage_data_type=storage_data_type,
                        padded=storage_data_type == "bfloat16",
                    ),
                )
                provider.compile_request(
                    reduction_request, reduction_baseline, "libtriton_jit"
                )
                reduction_manifest = json.loads(
                    (reduction_baseline / "manifest.json").read_bytes()
                )
                stage = reduction_manifest["program"]["stages"][0]
                assert stage["kernel_family"] == "reduction"
                assert stage["kernel"]["source"] == "reduction.py"
                assert stage["candidates"][0]["payload"]["meta"][
                    "REDUCTION_MODE"
                ] == REDUCTION_MODES[operation]
                _run_driver(
                    driver,
                    target,
                    reduction_request,
                    reduction_baseline,
                    "success",
                    f"valid_{case}",
                )
                reduction_fixtures[case] = (
                    reduction_request,
                    reduction_baseline,
                    reduction_manifest,
                )

        scalar_reduction_request = fixture_root / "request-reduction-scalar.json"
        scalar_reduction_baseline = fixture_root / "baseline-reduction-scalar"
        _write_request(
            scalar_reduction_request,
            _reduction_request(
                identity["identity_sha256"], target, scalar_output=True
            ),
        )
        provider.compile_request(
            scalar_reduction_request,
            scalar_reduction_baseline,
            "libtriton_jit",
        )
        _run_driver(
            driver,
            target,
            scalar_reduction_request,
            scalar_reduction_baseline,
            "success",
            "valid_reduction_rank0_output",
        )

        mixed_reduction_request = fixture_root / "request-mixed-reduction.json"
        mixed_reduction_baseline = fixture_root / "baseline-mixed-reduction"
        _write_request(
            mixed_reduction_request,
            _mixed_reduction_layout_pointwise_request(
                identity["identity_sha256"], target
            ),
        )
        provider.compile_request(
            mixed_reduction_request,
            mixed_reduction_baseline,
            "libtriton_jit",
        )
        mixed_reduction_manifest = json.loads(
            (mixed_reduction_baseline / "manifest.json").read_bytes()
        )
        assert [
            stage["kernel_family"]
            for stage in mixed_reduction_manifest["program"]["stages"]
        ] == ["reduction", "layout", "unary"]
        assert mixed_reduction_manifest["workspace_size"] == 512
        _run_driver(
            driver,
            target,
            mixed_reduction_request,
            mixed_reduction_baseline,
            "success",
            "valid_reduction_layout_pointwise_virtual_dag",
        )

        reduction_request, reduction_baseline, reduction_manifest = (
            reduction_fixtures["reduction_sum_float32"]
        )
        reduction_family_tamper = fixture_root / "reduction_family_tamper"
        shutil.copytree(reduction_baseline, reduction_family_tamper)
        changed_reduction = copy.deepcopy(reduction_manifest)
        changed_reduction["program"]["stages"][0]["kernel_family"] = "binary"
        (reduction_family_tamper / "manifest.json").write_text(
            json.dumps(changed_reduction, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            reduction_request,
            reduction_family_tamper,
            "compilation_failed",
            "reduction_family_tamper",
        )

        def run_reduction_manifest_mutation(
            case_name: str,
            fixture_name: str,
            mutate: Callable[[dict[str, Any]], None],
        ) -> None:
            source_request, source_baseline, source_manifest = (
                reduction_fixtures[fixture_name]
            )
            artifact = fixture_root / case_name
            shutil.copytree(source_baseline, artifact)
            changed = copy.deepcopy(source_manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                source_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        def change_reduction_meta(
            document: dict[str, Any], name: str, value: int
        ) -> None:
            payload = _payload(document)
            common = [
                "RANK", "OUTPUT_RANK", "AXIS", "KEEP_DIMENSIONS", "OUTER",
                "REDUCTION_SIZE", "INNER", "OUTPUT_ELEMENTS", "REDUCTION_MODE",
            ]
            meta_names = list(common)
            if payload["entry_point"] == "reduction_strided_persistent_kernel":
                meta_names += [
                    f"{prefix}_{axis}"
                    for prefix in (
                        "INPUT_DIM", "INPUT_STRIDE", "OUTPUT_DIM", "OUTPUT_STRIDE"
                    )
                    for axis in range(8)
                ]
            meta_names += ["BLOCK_SIZE", "WORKER_COUNT"]
            payload["meta"][name] = value
            tokens = payload["full_signature"].split(",")
            tokens[3 + meta_names.index(name)] = str(value)
            payload["full_signature"] = ",".join(tokens)

        run_reduction_manifest_mutation(
            "reduction_mode_meta_tamper",
            "reduction_sum_float32",
            lambda document: change_reduction_meta(
                document, "REDUCTION_MODE", 1
            ),
        )
        run_reduction_manifest_mutation(
            "reduction_outer_meta_tamper",
            "reduction_sum_float32",
            lambda document: change_reduction_meta(document, "OUTER", 3),
        )
        run_reduction_manifest_mutation(
            "reduction_axis_meta_tamper",
            "reduction_sum_float32",
            lambda document: change_reduction_meta(document, "AXIS", 0),
        )
        run_reduction_manifest_mutation(
            "reduction_general_stride_meta_tamper",
            "reduction_sum_bfloat16",
            lambda document: change_reduction_meta(
                document, "INPUT_STRIDE_5", 21
            ),
        )
        run_reduction_manifest_mutation(
            "reduction_source_tamper",
            "reduction_sum_float32",
            lambda document: document["program"]["stages"][0]["kernel"].__setitem__(
                "source", "binary.py"
            ),
        )

        source_request, source_baseline, source_manifest = reduction_fixtures[
            "reduction_sum_float32"
        ]
        rewritten_source_artifact = fixture_root / "reduction_full_source_tamper"
        shutil.copytree(source_baseline, rewritten_source_artifact)
        changed_source_manifest = copy.deepcopy(source_manifest)
        kernel = changed_source_manifest["program"]["stages"][0]["kernel"]
        source_path = rewritten_source_artifact / kernel["materialized_source"]["file"]
        source_path.write_bytes(source_path.read_bytes() + b"\n# tampered\n")
        changed_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        changed_source_name = source_path.name[:-67] + changed_source_sha + ".py"
        source_path.rename(source_path.with_name(changed_source_name))
        kernel["source_sha256"] = changed_source_sha
        kernel["materialized_source"]["file"] = changed_source_name
        kernel["materialized_source"]["size"] += len(b"\n# tampered\n")
        kernel["materialized_source"]["sha256"] = changed_source_sha
        changed_source_manifest["source_sha256"] = hashlib.sha256(
            json.dumps([changed_source_sha], separators=(",", ":")).encode()
        ).hexdigest()
        for candidate in changed_source_manifest["program"]["stages"][0][
            "candidates"
        ]:
            candidate["payload"]["source_path"] = changed_source_name
            candidate["payload"]["source_sha256"] = changed_source_sha
        (rewritten_source_artifact / "manifest.json").write_text(
            json.dumps(changed_source_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            source_request,
            rewritten_source_artifact,
            "compilation_failed",
            "reduction_full_source_tamper",
        )
        run_reduction_manifest_mutation(
            "reduction_fast_general_path_tamper",
            "reduction_sum_float32",
            lambda document: document["program"]["stages"][0]["kernel"].__setitem__(
                "entry_point", "reduction_strided_persistent_kernel"
            ),
        )

        def mutate_reduction_signature(document: dict[str, Any]) -> None:
            payload = _payload(document)
            tokens = payload["full_signature"].split(",")
            tokens[0] = "*fp16:16"
            payload["full_signature"] = ",".join(tokens)

        run_reduction_manifest_mutation(
            "reduction_signature_tamper",
            "reduction_sum_float32",
            mutate_reduction_signature,
        )
        run_reduction_manifest_mutation(
            "reduction_grid_tamper",
            "reduction_sum_float32",
            lambda document: _payload(document).__setitem__("grid", [1, 1, 1]),
        )
        run_reduction_manifest_mutation(
            "reduction_candidate_identity_tamper",
            "reduction_sum_float32",
            lambda document: document["program"]["stages"][0]["autotune"].__setitem__(
                "candidate_identity", "0" * 64
            ),
        )

        rank0_scalar_tamper = fixture_root / "reduction_rank0_scalar_tamper"
        shutil.copytree(scalar_reduction_baseline, rank0_scalar_tamper)
        rank0_manifest = json.loads(
            (scalar_reduction_baseline / "manifest.json").read_bytes()
        )
        rank0_manifest["program"]["stages"][0]["argument_sources"][-1][
            "value"
        ] = 2
        (rank0_scalar_tamper / "manifest.json").write_text(
            json.dumps(rank0_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver, target, scalar_reduction_request, rank0_scalar_tamper,
            "compilation_failed", "reduction_rank0_scalar_tamper",
        )

        mixed_dependency_tamper = fixture_root / "reduction_mixed_dependency_tamper"
        shutil.copytree(mixed_reduction_baseline, mixed_dependency_tamper)
        changed_mixed = copy.deepcopy(mixed_reduction_manifest)
        changed_mixed["program"]["stages"][1]["dependencies"] = []
        (mixed_dependency_tamper / "manifest.json").write_text(
            json.dumps(changed_mixed, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver, target, mixed_reduction_request, mixed_dependency_tamper,
            "compilation_failed", "reduction_mixed_dependency_tamper",
        )

        changed_graph = _reduction_request(
            identity["identity_sha256"], target, operation="reduction_avg"
        )
        changed_graph_request = fixture_root / "request-reduction-graph-tamper.json"
        changed_graph_hash = _write_request(changed_graph_request, changed_graph)
        changed_graph_artifact = fixture_root / "reduction_graph_attrs_tamper"
        shutil.copytree(reduction_baseline, changed_graph_artifact)
        changed_graph_manifest = copy.deepcopy(reduction_manifest)
        changed_graph_manifest["request_sha256"] = changed_graph_hash
        (changed_graph_artifact / "manifest.json").write_text(
            json.dumps(changed_graph_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver, target, changed_graph_request, changed_graph_artifact,
            "compilation_failed", "reduction_graph_attrs_tamper",
        )

        changed_mode = _reduction_request(
            identity["identity_sha256"], target, operation="reduction_sum"
        )
        changed_mode_graph = changed_mode["graph"]
        assert isinstance(changed_mode_graph, dict)
        changed_mode_node = changed_mode_graph["nodes"][0]
        assert isinstance(changed_mode_node, dict)
        changed_mode_attributes = changed_mode_node["attributes"]
        assert isinstance(changed_mode_attributes, dict)
        changed_mode_attributes["mode"] = REDUCTION_MODES["reduction_avg"]
        changed_mode_request = fixture_root / "request-reduction-mode-tamper.json"
        changed_mode_hash = _write_request(changed_mode_request, changed_mode)
        changed_mode_artifact = fixture_root / "reduction_graph_mode_tamper"
        shutil.copytree(reduction_baseline, changed_mode_artifact)
        changed_mode_manifest = copy.deepcopy(reduction_manifest)
        changed_mode_manifest["request_sha256"] = changed_mode_hash
        (changed_mode_artifact / "manifest.json").write_text(
            json.dumps(changed_mode_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver, target, changed_mode_request, changed_mode_artifact,
            "compilation_failed", "reduction_graph_mode_tamper",
        )

        layout_fixtures: dict[
            tuple[str, str],
            tuple[Path, Path, dict[str, object], dict[str, Any]],
        ] = {}
        layout_pointer_tokens = {
            "float32": "*i32:16",
            "float16": "*i16:16",
            "bfloat16": "*i16:16",
        }
        for operation in LAYOUT_OPERATIONS:
            for storage_data_type, pointer_token in layout_pointer_tokens.items():
                layout_document = _layout_request(
                    identity["identity_sha256"],
                    target,
                    operation=operation,
                    storage_data_type=storage_data_type,
                )
                layout_request = fixture_root / (
                    f"request-layout-{operation}-{storage_data_type}.json"
                )
                layout_baseline = fixture_root / (
                    f"baseline-layout-{operation}-{storage_data_type}"
                )
                _write_request(layout_request, layout_document)
                provider.compile_request(
                    layout_request, layout_baseline, "libtriton_jit"
                )
                layout_manifest = json.loads(
                    (layout_baseline / "manifest.json").read_bytes()
                )
                layout_stage = layout_manifest["program"]["stages"][0]
                assert layout_stage["kernel_family"] == "layout"
                assert layout_stage["kernel"]["source"] == "layout.py"
                assert layout_stage["kernel"]["entry_point"] == (
                    "layout_copy_kernel"
                )
                for candidate in layout_stage["candidates"]:
                    payload = candidate["payload"]
                    assert payload["argument_abi"] == [
                        {"index": 0, "name": "input_ptr", "type": "pointer"},
                        {"index": 1, "name": "output_ptr", "type": "pointer"},
                        {"index": 2, "name": "n_elements", "type": "i32"},
                    ]
                    assert payload["full_signature"].split(",")[:3] == [
                        pointer_token,
                        pointer_token,
                        "i32",
                    ]
                _run_driver(
                    driver,
                    target,
                    layout_request,
                    layout_baseline,
                    "success",
                    f"valid_layout_{operation}_{storage_data_type}",
                )
                layout_fixtures[(operation, storage_data_type)] = (
                    layout_request,
                    layout_baseline,
                    layout_document,
                    layout_manifest,
                )

        optimized_layout_fixtures = {}
        for operation, mode, grid in (
            ("reshape", 1, [1, 1, 1]),
            ("transpose", 1, [1, 1, 1]),
            ("slice", 2, [2, 1, 1]),
        ):
            optimized_document = _optimized_layout_request(
                identity["identity_sha256"], target, operation
            )
            optimized_request = fixture_root / f"request-layout-{operation}-optimized.json"
            optimized_baseline = fixture_root / f"baseline-layout-{operation}-optimized"
            _write_request(optimized_request, optimized_document)
            provider.compile_request(
                optimized_request, optimized_baseline, "libtriton_jit"
            )
            optimized_manifest = json.loads(
                (optimized_baseline / "manifest.json").read_bytes()
            )
            for candidate in optimized_manifest["program"]["stages"][0]["candidates"]:
                payload = candidate["payload"]
                assert payload["meta"]["LAYOUT_MODE"] == mode
                assert payload["grid"] == grid
            _run_driver(
                driver, target, optimized_request, optimized_baseline,
                "success", f"valid_layout_{operation}_optimized",
            )
            optimized_layout_fixtures[operation] = (
                optimized_request, optimized_baseline, optimized_manifest
            )

        scalar_layout_request = fixture_root / "request-layout-scalar.json"
        scalar_layout_baseline = fixture_root / "baseline-layout-scalar"
        _write_request(
            scalar_layout_request,
            _scalar_reshape_request(identity["identity_sha256"], target),
        )
        provider.compile_request(
            scalar_layout_request,
            scalar_layout_baseline,
            "libtriton_jit",
        )
        _run_driver(
            driver,
            target,
            scalar_layout_request,
            scalar_layout_baseline,
            "success",
            "valid_layout_scalar_reshape",
        )

        max_step_slice_document = _layout_request(
            identity["identity_sha256"], target, operation="slice"
        )
        max_step_slice_graph = max_step_slice_document["graph"]
        assert isinstance(max_step_slice_graph, dict)
        max_step_slice_graph["tensors"] = [
            _tensor(1, [6], [1]),
            _tensor(2, [1], [1]),
        ]
        max_step_slice_node = max_step_slice_graph["nodes"][0]
        assert isinstance(max_step_slice_node, dict)
        max_step_slice_node["attributes"] = {
            "n_elements": 1,
            "rank": 1,
            "starts": [0],
            "limits": [6],
            "slice_strides": [2**63 - 1],
            "input_dimensions": [6],
            "input_strides": [1],
            "output_dimensions": [1],
            "output_strides": [1],
        }
        max_step_slice_request = fixture_root / "request-layout-max-step.json"
        max_step_slice_baseline = fixture_root / "baseline-layout-max-step"
        _write_request(max_step_slice_request, max_step_slice_document)
        provider.compile_request(
            max_step_slice_request,
            max_step_slice_baseline,
            "libtriton_jit",
        )
        _run_driver(
            driver,
            target,
            max_step_slice_request,
            max_step_slice_baseline,
            "success",
            "valid_layout_slice_max_step",
        )

        mixed_layout_document = _mixed_layout_pointwise_request(
            identity["identity_sha256"], target
        )
        mixed_layout_request = fixture_root / "request-mixed-layout.json"
        mixed_layout_baseline = fixture_root / "baseline-mixed-layout"
        _write_request(mixed_layout_request, mixed_layout_document)
        provider.compile_request(
            mixed_layout_request,
            mixed_layout_baseline,
            "libtriton_jit",
        )
        mixed_layout_manifest = json.loads(
            (mixed_layout_baseline / "manifest.json").read_bytes()
        )
        assert [
            stage["kernel_family"]
            for stage in mixed_layout_manifest["program"]["stages"]
        ] == ["layout", "unary", "layout", "layout", "unary"]
        assert [
            stage["dependencies"]
            for stage in mixed_layout_manifest["program"]["stages"]
        ] == [[], [0], [1], [2], [3]]
        assert mixed_layout_manifest["workspace_size"] == 1024
        _run_driver(
            driver,
            target,
            mixed_layout_request,
            mixed_layout_baseline,
            "success",
            "valid_mixed_layout_pointwise_virtual_dag",
        )

        _, slice_baseline, slice_document, slice_manifest = layout_fixtures[
            ("slice", "float32")
        ]

        def run_layout_manifest_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(slice_baseline, artifact)
            changed = copy.deepcopy(slice_manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                layout_fixtures[("slice", "float32")][0],
                artifact,
                "compilation_failed",
                case_name,
            )

        def run_optimized_layout_manifest_mutation(
            operation: str,
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
        ) -> None:
            request, baseline, manifest = optimized_layout_fixtures[operation]
            artifact = fixture_root / case_name
            shutil.copytree(baseline, artifact)
            changed = copy.deepcopy(manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                request,
                artifact,
                "compilation_failed",
                case_name,
            )

        run_optimized_layout_manifest_mutation(
            "reshape",
            "layout_linear_mode_tamper",
            lambda document: _set_stage_meta(document, 0, "LAYOUT_MODE", 0),
        )
        run_optimized_layout_manifest_mutation(
            "slice",
            "layout_shared_row_mode_tamper",
            lambda document: _set_stage_meta(document, 0, "LAYOUT_MODE", 0),
        )

        def mutate_shared_row_grid(document: dict[str, Any]) -> None:
            _payload(document)["grid"] = [1, 1, 1]

        run_optimized_layout_manifest_mutation(
            "slice", "layout_shared_row_grid_tamper", mutate_shared_row_grid
        )

        run_layout_manifest_mutation(
            "layout_kernel_family_tamper",
            lambda document: document["program"]["stages"][0].__setitem__(
                "kernel_family", "unary"
            ),
        )
        run_layout_manifest_mutation(
            "layout_platform_source_tamper",
            lambda document: document["program"]["stages"][0]["kernel"].__setitem__(
                "source", "unary.py"
            ),
        )

        def mutate_layout_input_base(document: dict[str, Any]) -> None:
            _set_stage_meta(document, 0, "INPUT_BASE", 8)

        def mutate_layout_output_stride(document: dict[str, Any]) -> None:
            _set_stage_meta(document, 0, "OUTPUT_STRIDE_7", 2)

        def mutate_layout_element_size(document: dict[str, Any]) -> None:
            _set_stage_meta(document, 0, "ELEMENT_SIZE_BYTES", 2)

        def mutate_layout_argument_name(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["argument_sources"][0][
                "name"
            ] = "in_ptr"

        def mutate_layout_signature(document: dict[str, Any]) -> None:
            payload = _payload(document)
            tokens = payload["full_signature"].split(",")
            tokens[0] = "*fp16:16"
            payload["full_signature"] = ",".join(tokens)

        def mutate_layout_candidate_identity(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["autotune"][
                "candidate_identity"
            ] = "0" * 64

        run_layout_manifest_mutation(
            "layout_input_base_tamper", mutate_layout_input_base
        )
        run_layout_manifest_mutation(
            "layout_output_stride_tamper", mutate_layout_output_stride
        )
        run_layout_manifest_mutation(
            "layout_element_size_tamper", mutate_layout_element_size
        )
        run_layout_manifest_mutation(
            "layout_argument_name_tamper", mutate_layout_argument_name
        )
        run_layout_manifest_mutation(
            "layout_signature_tamper", mutate_layout_signature
        )
        run_layout_manifest_mutation(
            "layout_candidate_identity_tamper",
            mutate_layout_candidate_identity,
        )

        changed_slice_document = copy.deepcopy(slice_document)
        changed_slice_document["graph"]["nodes"][0]["attributes"]["starts"] = [
            1,
            0,
        ]
        changed_slice_document["graph"]["nodes"][0]["attributes"]["limits"] = [
            5,
            5,
        ]
        changed_slice_request = fixture_root / "request-layout-graph-tamper.json"
        changed_slice_hash = _write_request(
            changed_slice_request, changed_slice_document
        )
        changed_slice_artifact = fixture_root / "layout_graph_attrs_tamper"
        shutil.copytree(slice_baseline, changed_slice_artifact)
        changed_slice_manifest = copy.deepcopy(slice_manifest)
        changed_slice_manifest["request_sha256"] = changed_slice_hash
        (changed_slice_artifact / "manifest.json").write_text(
            json.dumps(changed_slice_manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            changed_slice_request,
            changed_slice_artifact,
            "compilation_failed",
            "layout_graph_attrs_tamper",
        )

        comparison_fixtures: dict[
            tuple[str, str],
            tuple[Path, Path, dict[str, object], dict[str, Any]],
        ] = {}
        comparison_pointer_tokens = {
            "float32": "*fp32:16",
            "float16": "*fp16:16",
            "bfloat16": "*bf16:16",
        }
        for operation, mode in COMPARISON_POINTWISE_MODES.items():
            for input_data_type, pointer_token in comparison_pointer_tokens.items():
                comparison_document = _comparison_request(
                    identity["identity_sha256"],
                    target,
                    operation=operation,
                    input_data_type=input_data_type,
                )
                comparison_request = fixture_root / (
                    f"request-{operation}-{input_data_type}.json"
                )
                comparison_baseline = fixture_root / (
                    f"baseline-{operation}-{input_data_type}"
                )
                _write_request(comparison_request, comparison_document)
                provider.compile_request(
                    comparison_request, comparison_baseline, "libtriton_jit"
                )
                comparison_manifest = json.loads(
                    (comparison_baseline / "manifest.json").read_bytes()
                )
                comparison_stage = comparison_manifest["program"]["stages"][0]
                assert comparison_stage["operation"] == operation
                assert comparison_stage["kernel"]["entry_point"] == (
                    "binary_strided_kernel"
                )
                for candidate in comparison_stage["candidates"]:
                    assert candidate["payload"]["meta"]["OP_KIND"] == mode
                    assert candidate["payload"]["full_signature"].split(",")[
                        :4
                    ] == [pointer_token, pointer_token, "*i8:16", "i32"]
                _run_driver(
                    driver,
                    target,
                    comparison_request,
                    comparison_baseline,
                    "success",
                    f"valid_comparison_{operation}_{input_data_type}_strided_autotune",
                )
                comparison_fixtures[(operation, input_data_type)] = (
                    comparison_request,
                    comparison_baseline,
                    comparison_document,
                    comparison_manifest,
                )

        mixed_comparison_document = _mixed_comparison_logical_request(
            identity["identity_sha256"], target
        )
        mixed_comparison_request = fixture_root / "request-mixed-comparison.json"
        mixed_comparison_baseline = fixture_root / "baseline-mixed-comparison"
        _write_request(mixed_comparison_request, mixed_comparison_document)
        provider.compile_request(
            mixed_comparison_request,
            mixed_comparison_baseline,
            "libtriton_jit",
        )
        mixed_comparison_manifest = json.loads(
            (mixed_comparison_baseline / "manifest.json").read_bytes()
        )
        mixed_comparison_stages = mixed_comparison_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_comparison_stages] == [
            "cmp_eq",
            "logical_not",
        ]
        assert [stage["dependencies"] for stage in mixed_comparison_stages] == [
            [],
            [0],
        ]
        assert mixed_comparison_manifest["workspace_size"] == 256
        assert mixed_comparison_stages[0]["candidates"][0]["payload"][
            "full_signature"
        ].split(",")[:4] == ["*fp32:16", "*fp32:16", "*i8:16", "i32"]
        assert mixed_comparison_stages[1]["candidates"][0]["payload"][
            "full_signature"
        ].split(",")[:3] == ["*i8:16", "*i8:16", "i32"]
        _run_driver(
            driver,
            target,
            mixed_comparison_request,
            mixed_comparison_baseline,
            "success",
            "valid_mixed_comparison_logical_bool_dag",
        )

        def run_comparison_request_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            _, base_artifact, base_document, base_manifest = comparison_fixtures[
                ("cmp_eq", "float32")
            ]
            artifact = fixture_root / case_name
            shutil.copytree(base_artifact, artifact)
            changed_request = copy.deepcopy(base_document)
            mutate(changed_request)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(changed_request_path, changed_request)
            changed_manifest = copy.deepcopy(base_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (artifact / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                artifact,
                "compilation_failed",
                case_name,
            )

        run_comparison_request_mutation(
            "comparison_float_output_rejected",
            lambda document: document["graph"]["tensors"][2].__setitem__(
                "data_type", "float32"
            ),
        )
        run_comparison_request_mutation(
            "comparison_boolean_input_rejected",
            lambda document: document["graph"]["tensors"][0].__setitem__(
                "data_type", "boolean"
            ),
        )
        run_comparison_request_mutation(
            "comparison_float_compute_rejected",
            lambda document: document["graph"]["nodes"][0].__setitem__(
                "compute_data_type", "float32"
            ),
        )
        comparison_signature_artifact = fixture_root / "comparison_signature_tamper"
        shutil.copytree(
            comparison_fixtures[("cmp_eq", "float32")][1],
            comparison_signature_artifact,
        )
        comparison_signature_manifest = copy.deepcopy(
            comparison_fixtures[("cmp_eq", "float32")][3]
        )
        for candidate in comparison_signature_manifest["program"]["stages"][0][
            "candidates"
        ]:
            tokens = candidate["payload"]["full_signature"].split(",")
            tokens[2] = "*fp32:16"
            candidate["payload"]["full_signature"] = ",".join(tokens)
        (comparison_signature_artifact / "manifest.json").write_text(
            json.dumps(comparison_signature_manifest, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            comparison_fixtures[("cmp_eq", "float32")][0],
            comparison_signature_artifact,
            "compilation_failed",
            "comparison_signature_tamper",
        )

        binary_select_fixtures: dict[
            str, tuple[Path, Path, dict[str, object], dict[str, Any]]
        ] = {}
        for storage_data_type, pointer_token in comparison_pointer_tokens.items():
            select_document = _binary_select_request(
                identity["identity_sha256"],
                target,
                storage_data_type=storage_data_type,
            )
            select_request = fixture_root / (
                f"request-binary-select-{storage_data_type}.json"
            )
            select_baseline = fixture_root / (
                f"baseline-binary-select-{storage_data_type}"
            )
            _write_request(select_request, select_document)
            provider.compile_request(
                select_request, select_baseline, "libtriton_jit"
            )
            select_manifest = json.loads(
                (select_baseline / "manifest.json").read_bytes()
            )
            select_stage = select_manifest["program"]["stages"][0]
            assert select_stage["operation"] == "binary_select"
            assert select_stage["kernel"]["source"] == "ternary.py"
            assert select_stage["kernel"]["entry_point"] == (
                "binary_select_strided_kernel"
            )
            assert [
                source["name"] for source in select_stage["argument_sources"]
            ] == ["x_ptr", "y_ptr", "t_ptr", "out_ptr", "n_elements"]
            assert select_stage["autotune"]["enabled"] is True
            assert len(select_stage["candidates"]) == 2
            for candidate in select_stage["candidates"]:
                payload = candidate["payload"]
                assert payload["full_signature"].split(",")[:5] == [
                    pointer_token,
                    pointer_token,
                    "*i8:16",
                    pointer_token,
                    "i32",
                ]
                assert payload["meta"]["MASK_STRIDE_5"] == 4
                assert payload["meta"]["MASK_STRIDE_6"] == 1
                assert payload["meta"]["MASK_STRIDE_7"] == 0
                assert "OP_KIND" not in payload["meta"]
                assert "ALPHA" not in payload["meta"]
            _run_driver(
                driver,
                target,
                select_request,
                select_baseline,
                "success",
                f"valid_binary_select_{storage_data_type}_mixed_abi_strided_autotune",
            )
            binary_select_fixtures[storage_data_type] = (
                select_request,
                select_baseline,
                select_document,
                select_manifest,
            )

        mixed_select_document = _mixed_binary_select_request(
            identity["identity_sha256"], target
        )
        mixed_select_request = fixture_root / "request-mixed-binary-select.json"
        mixed_select_baseline = fixture_root / "baseline-mixed-binary-select"
        _write_request(mixed_select_request, mixed_select_document)
        provider.compile_request(
            mixed_select_request, mixed_select_baseline, "libtriton_jit"
        )
        mixed_select_manifest = json.loads(
            (mixed_select_baseline / "manifest.json").read_bytes()
        )
        mixed_select_stages = mixed_select_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_select_stages] == [
            "cmp_gt",
            "binary_select",
            "relu",
        ]
        assert [stage["dependencies"] for stage in mixed_select_stages] == [
            [],
            [0],
            [1],
        ]
        assert mixed_select_manifest["workspace_size"] == 512
        assert mixed_select_stages[1]["argument_sources"][2]["source"] == (
            "graph_workspace"
        )
        assert mixed_select_stages[1]["argument_sources"][2]["size"] == 7
        assert mixed_select_stages[1]["argument_sources"][3]["source"] == (
            "graph_workspace"
        )
        assert mixed_select_stages[1]["argument_sources"][3]["size"] == 32
        _run_driver(
            driver,
            target,
            mixed_select_request,
            mixed_select_baseline,
            "success",
            "valid_mixed_binary_select_virtual_dag",
        )

        def run_binary_select_request_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            _, base_artifact, base_document, base_manifest = (
                binary_select_fixtures["float32"]
            )
            artifact = fixture_root / case_name
            shutil.copytree(base_artifact, artifact)
            changed_request = copy.deepcopy(base_document)
            mutate(changed_request)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(changed_request_path, changed_request)
            changed_manifest = copy.deepcopy(base_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (artifact / "manifest.json").write_text(
                json.dumps(changed_manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                artifact,
                "compilation_failed",
                case_name,
            )

        run_binary_select_request_mutation(
            "binary_select_value_dtype_tamper",
            lambda document: document["graph"]["tensors"][1].__setitem__(
                "data_type", "float16"
            ),
        )
        run_binary_select_request_mutation(
            "binary_select_predicate_dtype_tamper",
            lambda document: document["graph"]["tensors"][2].__setitem__(
                "data_type", "float32"
            ),
        )
        run_binary_select_request_mutation(
            "binary_select_output_dtype_tamper",
            lambda document: document["graph"]["tensors"][3].__setitem__(
                "data_type", "float16"
            ),
        )
        run_binary_select_request_mutation(
            "binary_select_compute_dtype_tamper",
            lambda document: document["graph"]["nodes"][0].__setitem__(
                "compute_data_type", "boolean"
            ),
        )
        run_binary_select_request_mutation(
            "binary_select_port_name_tamper",
            lambda document: document["graph"]["nodes"][0]["inputs"][2].__setitem__(
                "name", "mask"
            ),
        )
        run_binary_select_request_mutation(
            "binary_select_mode_tamper",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("mode", 40),
        )

        select_request, select_baseline, _, select_manifest = (
            binary_select_fixtures["float32"]
        )

        def run_binary_select_manifest_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(select_baseline, artifact)
            changed = copy.deepcopy(select_manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                select_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        def mutate_select_signature(document: dict[str, Any]) -> None:
            for candidate in document["program"]["stages"][0]["candidates"]:
                tokens = candidate["payload"]["full_signature"].split(",")
                tokens[2] = "*fp32:16"
                candidate["payload"]["full_signature"] = ",".join(tokens)

        def mutate_select_mask_stride(document: dict[str, Any]) -> None:
            for candidate in document["program"]["stages"][0]["candidates"]:
                payload = candidate["payload"]
                payload["meta"]["MASK_STRIDE_5"] += 1
                tokens = payload["full_signature"].split(",")
                tokens[34] = str(payload["meta"]["MASK_STRIDE_5"])
                payload["full_signature"] = ",".join(tokens)

        def mutate_select_argument_name(document: dict[str, Any]) -> None:
            stage = document["program"]["stages"][0]
            stage["argument_sources"][2]["name"] = "mask_ptr"
            for candidate in stage["candidates"]:
                candidate["payload"]["argument_abi"][2]["name"] = "mask_ptr"

        run_binary_select_manifest_mutation(
            "binary_select_signature_tamper", mutate_select_signature
        )
        run_binary_select_manifest_mutation(
            "binary_select_mask_stride_tamper", mutate_select_mask_stride
        )
        run_binary_select_manifest_mutation(
            "binary_select_argument_name_tamper", mutate_select_argument_name
        )
        run_binary_select_manifest_mutation(
            "binary_select_source_tamper",
            lambda document: document["program"]["stages"][0]["kernel"].__setitem__(
                "source", "binary.py"
            ),
        )
        run_binary_select_manifest_mutation(
            "binary_select_candidate_identity_tamper",
            lambda document: document["program"]["stages"][0]["autotune"].__setitem__(
                "candidate_identity", "0" * 64
            ),
        )

        logical_fixtures: dict[
            str,
            tuple[Path, Path, dict[str, object], dict[str, Any]],
        ] = {}
        logical_request_documents = {
            "logical_not": _logical_not_request(
                identity["identity_sha256"], target
            ),
            "logical_and": _logical_binary_request(
                identity["identity_sha256"],
                target,
                operation="logical_and",
            ),
            "logical_or": _logical_binary_request(
                identity["identity_sha256"],
                target,
                operation="logical_or",
            ),
        }
        for operation, logical_document in logical_request_documents.items():
            logical_request = fixture_root / f"request-{operation}.json"
            logical_baseline = fixture_root / f"baseline-{operation}"
            _write_request(logical_request, logical_document)
            provider.compile_request(
                logical_request, logical_baseline, "libtriton_jit"
            )
            logical_manifest = json.loads(
                (logical_baseline / "manifest.json").read_bytes()
            )
            logical_stage = logical_manifest["program"]["stages"][0]
            logical_unary = operation == "logical_not"
            expected_source = "unary.py" if logical_unary else "binary.py"
            expected_entry_point = (
                "unary_pointwise_strided_kernel"
                if logical_unary
                else "binary_strided_kernel"
            )
            expected_pointer_count = 2 if logical_unary else 3
            expected_mode_name = "OPERATION" if logical_unary else "OP_KIND"
            expected_mode = (
                LOGICAL_UNARY_POINTWISE_MODES[operation]
                if logical_unary
                else LOGICAL_BINARY_POINTWISE_MODES[operation]
            )
            assert logical_stage["operation"] == operation
            assert logical_stage["kernel"]["source"] == expected_source
            assert logical_stage["kernel"]["entry_point"] == expected_entry_point
            assert len(logical_stage["argument_sources"]) == (
                expected_pointer_count + 1
            )
            assert logical_stage["autotune"]["enabled"] is True
            assert logical_stage["autotune"]["selection"] == {
                "state": "pending",
                "candidate_id": "",
            }
            assert [
                candidate["payload"]["meta"]["BLOCK_SIZE"]
                for candidate in logical_stage["candidates"]
            ] == [256, 128]
            for candidate in logical_stage["candidates"]:
                payload = candidate["payload"]
                assert payload["meta"][expected_mode_name] == expected_mode
                assert payload["full_signature"].split(",")[
                    :expected_pointer_count
                ] == ["*i8:16"] * expected_pointer_count
            _run_driver(
                driver,
                target,
                logical_request,
                logical_baseline,
                "success",
                f"valid_{operation}_boolean_strided_autotune",
            )
            logical_fixtures[operation] = (
                logical_request,
                logical_baseline,
                logical_document,
                logical_manifest,
            )

        assert len(
            {
                fixture[3]["program"]["stages"][0]["autotune"][
                    "candidate_identity"
                ]
                for fixture in logical_fixtures.values()
            }
        ) == len(logical_fixtures)

        mixed_logical_document = _mixed_logical_request(
            identity["identity_sha256"], target
        )
        mixed_logical_request = fixture_root / "request-mixed-logical.json"
        mixed_logical_baseline = fixture_root / "baseline-mixed-logical"
        _write_request(mixed_logical_request, mixed_logical_document)
        provider.compile_request(
            mixed_logical_request,
            mixed_logical_baseline,
            "libtriton_jit",
        )
        mixed_logical_manifest = json.loads(
            (mixed_logical_baseline / "manifest.json").read_bytes()
        )
        mixed_logical_stages = mixed_logical_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_logical_stages] == [
            "logical_and",
            "logical_not",
            "logical_or",
        ]
        assert [stage["dependencies"] for stage in mixed_logical_stages] == [
            [],
            [0],
            [1],
        ]
        assert [stage["kernel"]["source"] for stage in mixed_logical_stages] == [
            "binary.py",
            "unary.py",
            "binary.py",
        ]
        assert mixed_logical_manifest["workspace_size"] == 512
        for stage, pointer_count in zip(
            mixed_logical_stages, (3, 2, 3), strict=True
        ):
            assert len(stage["candidates"]) == 2
            assert stage["autotune"]["enabled"] is True
            for candidate in stage["candidates"]:
                assert candidate["payload"]["full_signature"].split(",")[
                    :pointer_count
                ] == ["*i8:16"] * pointer_count
        _run_driver(
            driver,
            target,
            mixed_logical_request,
            mixed_logical_baseline,
            "success",
            "valid_mixed_logical_bool_dag",
        )
        logical_fixtures["mixed_logical"] = (
            mixed_logical_request,
            mixed_logical_baseline,
            mixed_logical_document,
            mixed_logical_manifest,
        )

        def write_logical_manifest_mutation(
            operation: str,
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
        ) -> None:
            base_request, base_artifact, _, base_manifest = logical_fixtures[
                operation
            ]
            artifact = fixture_root / case_name
            shutil.copytree(base_artifact, artifact)
            changed_manifest = copy.deepcopy(base_manifest)
            mutate(changed_manifest)
            (artifact / "manifest.json").write_text(
                json.dumps(
                    changed_manifest,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                base_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        def write_logical_request_mutation(
            operation: str,
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
        ) -> None:
            _, base_artifact, base_document, base_manifest = logical_fixtures[
                operation
            ]
            artifact = fixture_root / case_name
            shutil.copytree(base_artifact, artifact)
            changed_request = copy.deepcopy(base_document)
            mutate(changed_request)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(
                changed_request_path, changed_request
            )
            changed_manifest = copy.deepcopy(base_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (artifact / "manifest.json").write_text(
                json.dumps(
                    changed_manifest,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                artifact,
                "compilation_failed",
                case_name,
            )

        def mutate_logical_i1_signature(document: dict[str, Any]) -> None:
            payload = _payload(document)
            tokens = payload["full_signature"].split(",")
            tokens[0] = "*i1:16"
            payload["full_signature"] = ",".join(tokens)

        write_logical_manifest_mutation(
            "logical_not",
            "logical_not_i1_signature_tamper",
            mutate_logical_i1_signature,
        )
        write_logical_manifest_mutation(
            "logical_not",
            "logical_not_mode_tamper",
            lambda document: _set_stage_meta(document, 0, "OPERATION", 23),
        )
        write_logical_manifest_mutation(
            "logical_not",
            "logical_not_source_tamper",
            lambda document: document["program"]["stages"][0][
                "kernel"
            ].__setitem__("source", "binary.py"),
        )
        write_logical_manifest_mutation(
            "logical_and",
            "logical_and_argument_source_tamper",
            lambda document: document["program"]["stages"][0][
                "argument_sources"
            ][0].__setitem__("uid", 2),
        )
        write_logical_manifest_mutation(
            "logical_or",
            "logical_or_candidate_identity_tamper",
            lambda document: document["program"]["stages"][0]["autotune"].__setitem__(
                "candidate_identity", "0" * 64
            ),
        )
        write_logical_manifest_mutation(
            "mixed_logical",
            "mixed_logical_dependency_tamper",
            lambda document: document["program"]["stages"][1].__setitem__(
                "dependencies", []
            ),
        )

        def mutate_all_tensor_types(
            document: dict[str, Any], data_type: str
        ) -> None:
            for tensor in document["graph"]["tensors"]:
                tensor["data_type"] = data_type

        write_logical_request_mutation(
            "logical_not",
            "logical_not_float32_storage_rejected",
            lambda document: mutate_all_tensor_types(document, "float32"),
        )
        write_logical_request_mutation(
            "logical_not",
            "logical_not_float32_compute_rejected",
            lambda document: document["graph"]["nodes"][0].__setitem__(
                "compute_data_type", "float32"
            ),
        )
        write_logical_request_mutation(
            "logical_not",
            "logical_not_graph_mode_tamper",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("mode", 23),
        )

        def mutate_incompatible_logical_broadcast(
            document: dict[str, Any]
        ) -> None:
            left = document["graph"]["tensors"][0]
            left["dimensions"] = [2, 2]
            left["strides"] = [2, 1]

        write_logical_request_mutation(
            "logical_and",
            "logical_and_incompatible_broadcast_rejected",
            mutate_incompatible_logical_broadcast,
        )

        numeric_boolean_document = _request(
            identity["identity_sha256"], target
        )
        mutate_all_tensor_types(numeric_boolean_document, "boolean")
        for node in numeric_boolean_document["graph"]["nodes"]:
            node["compute_data_type"] = "boolean"
        numeric_boolean_request = fixture_root / "request-numeric-boolean.json"
        numeric_boolean_sha256 = _write_request(
            numeric_boolean_request, numeric_boolean_document
        )
        numeric_boolean_artifact = fixture_root / "numeric-boolean-rejected"
        shutil.copytree(baseline, numeric_boolean_artifact)
        numeric_boolean_manifest = json.loads(
            (numeric_boolean_artifact / "manifest.json").read_bytes()
        )
        numeric_boolean_manifest["request_sha256"] = numeric_boolean_sha256
        (numeric_boolean_artifact / "manifest.json").write_text(
            json.dumps(
                numeric_boolean_manifest,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            numeric_boolean_request,
            numeric_boolean_artifact,
            "compilation_failed",
            "numeric_boolean_storage_compute_rejected",
        )

        mixed_request_document = _mixed_binary_request(
            identity["identity_sha256"], target
        )
        mixed_request = fixture_root / "request-mixed-sub-mul.json"
        mixed_baseline = fixture_root / "baseline-mixed-sub-mul"
        _write_request(mixed_request, mixed_request_document)
        provider.compile_request(
            mixed_request, mixed_baseline, "libtriton_jit"
        )
        mixed_manifest = json.loads(
            (mixed_baseline / "manifest.json").read_bytes()
        )
        mixed_stages = mixed_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_stages] == ["sub", "mul"]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OP_KIND"]
            for stage in mixed_stages
        ] == [17, 18]
        assert [stage["dependencies"] for stage in mixed_stages] == [
            [],
            [0],
        ]
        _run_driver(
            driver,
            target,
            mixed_request,
            mixed_baseline,
            "success",
            "valid_mixed_sub_mul_dag",
        )

        sigmoid_backward_fixtures: dict[
            str, tuple[Path, Path, dict[str, object]]
        ] = {}
        for storage_data_type in ("float32", "float16", "bfloat16"):
            sigmoid_backward_document = _sigmoid_backward_request(
                identity["identity_sha256"],
                target,
                storage_data_type=storage_data_type,
            )
            sigmoid_backward_request = fixture_root / (
                f"request-sigmoid-backward-{storage_data_type}.json"
            )
            sigmoid_backward_baseline = fixture_root / (
                f"baseline-sigmoid-backward-{storage_data_type}"
            )
            _write_request(sigmoid_backward_request, sigmoid_backward_document)
            provider.compile_request(
                sigmoid_backward_request,
                sigmoid_backward_baseline,
                "libtriton_jit",
            )
            sigmoid_backward_manifest = json.loads(
                (sigmoid_backward_baseline / "manifest.json").read_bytes()
            )
            sigmoid_backward_stage = sigmoid_backward_manifest["program"][
                "stages"
            ][0]
            assert sigmoid_backward_stage["operation"] == "sigmoid_backward"
            assert sigmoid_backward_stage["kernel"]["source"] == "binary.py"
            assert sigmoid_backward_stage["kernel"]["entry_point"] == (
                "binary_strided_kernel"
            )
            assert len(sigmoid_backward_stage["argument_sources"]) == 4
            assert [
                candidate["payload"]["meta"]["BLOCK_SIZE"]
                for candidate in sigmoid_backward_stage["candidates"]
            ] == [256, 128]
            assert all(
                candidate["payload"]["meta"]["OP_KIND"] == 40
                for candidate in sigmoid_backward_stage["candidates"]
            )
            _run_driver(
                driver,
                target,
                sigmoid_backward_request,
                sigmoid_backward_baseline,
                "success",
                f"valid_sigmoid_backward_{storage_data_type}_strided_autotune",
            )
            sigmoid_backward_fixtures[storage_data_type] = (
                sigmoid_backward_request,
                sigmoid_backward_baseline,
                sigmoid_backward_document,
            )

        mixed_sigmoid_backward_document = _mixed_sigmoid_backward_request(
            identity["identity_sha256"], target
        )
        mixed_sigmoid_backward_request = fixture_root / (
            "request-mixed-sigmoid-backward-add.json"
        )
        mixed_sigmoid_backward_baseline = fixture_root / (
            "baseline-mixed-sigmoid-backward-add"
        )
        _write_request(
            mixed_sigmoid_backward_request, mixed_sigmoid_backward_document
        )
        provider.compile_request(
            mixed_sigmoid_backward_request,
            mixed_sigmoid_backward_baseline,
            "libtriton_jit",
        )
        mixed_sigmoid_backward_manifest = json.loads(
            (mixed_sigmoid_backward_baseline / "manifest.json").read_bytes()
        )
        mixed_sigmoid_backward_stages = mixed_sigmoid_backward_manifest[
            "program"
        ]["stages"]
        assert [stage["operation"] for stage in mixed_sigmoid_backward_stages] == [
            "sigmoid_backward",
            "add",
        ]
        assert [stage["dependencies"] for stage in mixed_sigmoid_backward_stages] == [
            [],
            [0],
        ]
        _run_driver(
            driver,
            target,
            mixed_sigmoid_backward_request,
            mixed_sigmoid_backward_baseline,
            "success",
            "valid_mixed_sigmoid_backward_add_dag",
        )

        sigmoid_request, sigmoid_baseline, sigmoid_document = (
            sigmoid_backward_fixtures["float32"]
        )
        broadcast_document = copy.deepcopy(sigmoid_document)
        broadcast_document["graph"]["tensors"][0]["dimensions"] = [2, 1]
        broadcast_document["graph"]["tensors"][0]["strides"] = [1, 0]
        broadcast_request = fixture_root / "request-sigmoid-backward-broadcast.json"
        broadcast_sha256 = _write_request(broadcast_request, broadcast_document)
        broadcast_artifact = fixture_root / "sigmoid-backward-broadcast"
        shutil.copytree(sigmoid_baseline, broadcast_artifact)
        broadcast_manifest = json.loads(
            (broadcast_artifact / "manifest.json").read_bytes()
        )
        broadcast_manifest["request_sha256"] = broadcast_sha256
        (broadcast_artifact / "manifest.json").write_text(
            json.dumps(
                broadcast_manifest, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            broadcast_request,
            broadcast_artifact,
            "compilation_failed",
            "sigmoid_backward_graph_broadcast_rejected",
        )

        def write_sigmoid_backward_manifest_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(sigmoid_baseline, artifact)
            changed = json.loads((artifact / "manifest.json").read_bytes())
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                sigmoid_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        def mutate_sigmoid_backward_mode(document: dict[str, Any]) -> None:
            _set_stage_meta(document, 0, "OP_KIND", 23)

        def mutate_sigmoid_backward_type(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["operation"] = "pow"

        def mutate_sigmoid_backward_signature(document: dict[str, Any]) -> None:
            candidate = document["program"]["stages"][0]["candidates"][0]
            tokens = candidate["payload"]["full_signature"].split(",")
            tokens[-4] = "23"
            candidate["payload"]["full_signature"] = ",".join(tokens)

        def mutate_sigmoid_backward_source(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["kernel"]["source"] = "unary.py"

        def mutate_sigmoid_backward_candidate_identity(
            document: dict[str, Any]
        ) -> None:
            document["program"]["stages"][0]["autotune"][
                "candidate_identity"
            ] = "0" * 64

        for case_name, mutator in (
            ("sigmoid_backward_type_tamper", mutate_sigmoid_backward_type),
            ("sigmoid_backward_mode_tamper", mutate_sigmoid_backward_mode),
            (
                "sigmoid_backward_signature_tamper",
                mutate_sigmoid_backward_signature,
            ),
            ("sigmoid_backward_source_tamper", mutate_sigmoid_backward_source),
            (
                "sigmoid_backward_candidate_identity_tamper",
                mutate_sigmoid_backward_candidate_identity,
            ),
        ):
            write_sigmoid_backward_manifest_mutation(case_name, mutator)

        unary_fixtures: dict[str, tuple[Path, Path, dict[str, object]]] = {}
        for operation in UNARY_POINTWISE_MODES:
            semantic_attributes: dict[str, float] = {}
            if operation == "elu":
                semantic_attributes["elu_alpha"] = 1.25
            elif operation == "softplus":
                semantic_attributes["softplus_beta"] = 0.75
            elif operation == "swish":
                semantic_attributes["swish_beta"] = 0.5
            unary_request_document = _unary_request(
                identity["identity_sha256"],
                target,
                operation=operation,
                **semantic_attributes,
            )
            unary_request = fixture_root / f"request-unary-{operation}.json"
            unary_baseline = fixture_root / f"baseline-unary-{operation}"
            _write_request(unary_request, unary_request_document)
            provider.compile_request(
                unary_request, unary_baseline, "libtriton_jit"
            )
            unary_manifest = json.loads(
                (unary_baseline / "manifest.json").read_bytes()
            )
            unary_stage = unary_manifest["program"]["stages"][0]
            assert unary_stage["operation"] == operation
            assert unary_stage["kernel"]["source"] == "unary.py"
            assert unary_stage["kernel"]["entry_point"] == (
                "unary_pointwise_strided_kernel"
            )
            assert len(unary_stage["argument_sources"]) == 3
            assert len(unary_stage["candidates"]) == 2
            unary_meta = unary_stage["candidates"][0]["payload"]["meta"]
            assert unary_meta["SWISH_BETA"] == semantic_attributes.get(
                "swish_beta", 1.0
            )
            assert unary_meta["ELU_ALPHA"] == semantic_attributes.get(
                "elu_alpha", 1.0
            )
            assert unary_meta["SOFTPLUS_BETA"] == semantic_attributes.get(
                "softplus_beta", 1.0
            )
            _run_driver(
                driver,
                target,
                unary_request,
                unary_baseline,
                "success",
                f"valid_unary_{operation}_autotune",
            )
            unary_fixtures[operation] = (
                unary_request,
                unary_baseline,
                unary_request_document,
            )

        for operation in (
            "sqrt",
            "rsqrt",
            "reciprocal",
            "exp",
            "log",
            "tanh",
            "sigmoid",
            "elu",
            "softplus",
            "swish",
            "gelu",
            "gelu_approx_tanh",
            "sin",
            "cos",
            "tan",
            "erf",
        ):
            for storage_data_type in ("float16", "bfloat16"):
                semantic_attributes = {}
                if operation == "elu":
                    semantic_attributes["elu_alpha"] = 1.25
                elif operation == "softplus":
                    semantic_attributes["softplus_beta"] = 0.75
                elif operation == "swish":
                    semantic_attributes["swish_beta"] = 0.5
                typed_request = fixture_root / (
                    f"request-unary-{operation}-{storage_data_type}.json"
                )
                typed_baseline = fixture_root / (
                    f"baseline-unary-{operation}-{storage_data_type}"
                )
                _write_request(
                    typed_request,
                    _unary_request(
                        identity["identity_sha256"],
                        target,
                        operation=operation,
                        storage_data_type=storage_data_type,
                        **semantic_attributes,
                    ),
                )
                provider.compile_request(
                    typed_request, typed_baseline, "libtriton_jit"
                )
                _run_driver(
                    driver,
                    target,
                    typed_request,
                    typed_baseline,
                    "success",
                    f"valid_unary_{operation}_{storage_data_type}",
                )

        for operation, attribute, value in (
            ("elu", "elu_alpha", -0.5),
            ("swish", "swish_beta", 0.0),
        ):
            edge_request = fixture_root / (
                f"request-unary-{operation}-semantic-edge.json"
            )
            edge_baseline = fixture_root / (
                f"baseline-unary-{operation}-semantic-edge"
            )
            _write_request(
                edge_request,
                _unary_request(
                    identity["identity_sha256"],
                    target,
                    operation=operation,
                    **{attribute: value},
                ),
            )
            provider.compile_request(
                edge_request, edge_baseline, "libtriton_jit"
            )
            _run_driver(
                driver,
                target,
                edge_request,
                edge_baseline,
                "success",
                f"valid_unary_{operation}_semantic_edge",
            )

        mixed_pointwise_document = _mixed_pointwise_request(
            identity["identity_sha256"], target
        )
        mixed_pointwise_request = fixture_root / "request-mixed-pointwise.json"
        mixed_pointwise_baseline = fixture_root / "baseline-mixed-pointwise"
        _write_request(mixed_pointwise_request, mixed_pointwise_document)
        provider.compile_request(
            mixed_pointwise_request,
            mixed_pointwise_baseline,
            "libtriton_jit",
        )
        mixed_pointwise_manifest = json.loads(
            (mixed_pointwise_baseline / "manifest.json").read_bytes()
        )
        mixed_pointwise_stages = mixed_pointwise_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_pointwise_stages] == [
            "add",
            "relu",
            "mul",
        ]
        assert [stage["dependencies"] for stage in mixed_pointwise_stages] == [
            [],
            [0],
            [1],
        ]
        assert [
            stage["kernel"]["source"] for stage in mixed_pointwise_stages
        ] == [
            "binary.py",
            "unary.py",
            "binary.py",
        ]
        _run_driver(
            driver,
            target,
            mixed_pointwise_request,
            mixed_pointwise_baseline,
            "success",
            "valid_mixed_binary_unary_dag",
        )

        mixed_math_document = _mixed_unary_math_request(
            identity["identity_sha256"], target
        )
        mixed_math_request = fixture_root / "request-mixed-unary-math.json"
        mixed_math_baseline = fixture_root / "baseline-mixed-unary-math"
        _write_request(mixed_math_request, mixed_math_document)
        provider.compile_request(
            mixed_math_request, mixed_math_baseline, "libtriton_jit"
        )
        mixed_math_manifest = json.loads(
            (mixed_math_baseline / "manifest.json").read_bytes()
        )
        mixed_math_stages = mixed_math_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_math_stages] == [
            "sqrt",
            "rsqrt",
            "reciprocal",
        ]
        assert [stage["dependencies"] for stage in mixed_math_stages] == [
            [],
            [0],
            [1],
        ]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OPERATION"]
            for stage in mixed_math_stages
        ] == [3, 13, 16]
        _run_driver(
            driver,
            target,
            mixed_math_request,
            mixed_math_baseline,
            "success",
            "valid_mixed_sqrt_rsqrt_reciprocal_dag",
        )

        mixed_transcendental_document = _mixed_unary_math_request(
            identity["identity_sha256"],
            target,
            ("exp", "log", "tanh"),
        )
        mixed_transcendental_request = (
            fixture_root / "request-mixed-exp-log-tanh.json"
        )
        mixed_transcendental_baseline = (
            fixture_root / "baseline-mixed-exp-log-tanh"
        )
        _write_request(
            mixed_transcendental_request, mixed_transcendental_document
        )
        provider.compile_request(
            mixed_transcendental_request,
            mixed_transcendental_baseline,
            "libtriton_jit",
        )
        mixed_transcendental_manifest = json.loads(
            (mixed_transcendental_baseline / "manifest.json").read_bytes()
        )
        mixed_transcendental_stages = mixed_transcendental_manifest["program"][
            "stages"
        ]
        assert [
            stage["operation"] for stage in mixed_transcendental_stages
        ] == ["exp", "log", "tanh"]
        assert [
            stage["dependencies"] for stage in mixed_transcendental_stages
        ] == [[], [0], [1]]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OPERATION"]
            for stage in mixed_transcendental_stages
        ] == [6, 7, 34]
        _run_driver(
            driver,
            target,
            mixed_transcendental_request,
            mixed_transcendental_baseline,
            "success",
            "valid_mixed_exp_log_tanh_dag",
        )

        mixed_special_math_document = _mixed_unary_math_request(
            identity["identity_sha256"],
            target,
            ("sin", "cos", "tan", "erf"),
        )
        mixed_special_math_request = (
            fixture_root / "request-mixed-sin-cos-tan-erf.json"
        )
        mixed_special_math_baseline = (
            fixture_root / "baseline-mixed-sin-cos-tan-erf"
        )
        _write_request(
            mixed_special_math_request, mixed_special_math_document
        )
        provider.compile_request(
            mixed_special_math_request,
            mixed_special_math_baseline,
            "libtriton_jit",
        )
        mixed_special_math_manifest = json.loads(
            (mixed_special_math_baseline / "manifest.json").read_bytes()
        )
        mixed_special_math_stages = mixed_special_math_manifest["program"][
            "stages"
        ]
        assert [
            stage["operation"] for stage in mixed_special_math_stages
        ] == ["sin", "cos", "tan", "erf"]
        assert [
            stage["dependencies"] for stage in mixed_special_math_stages
        ] == [[], [0], [1], [2]]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OPERATION"]
            for stage in mixed_special_math_stages
        ] == [14, 11, 15, 4]
        assert [
            len(stage["argument_sources"])
            for stage in mixed_special_math_stages
        ] == [3, 3, 3, 3]
        assert mixed_special_math_manifest["workspace_size"] == 768
        mixed_activation_document = _mixed_activation_request(
            identity["identity_sha256"], target
        )
        mixed_activation_request = (
            fixture_root / "request-mixed-activation.json"
        )
        mixed_activation_baseline = fixture_root / "baseline-mixed-activation"
        _write_request(mixed_activation_request, mixed_activation_document)
        provider.compile_request(
            mixed_activation_request,
            mixed_activation_baseline,
            "libtriton_jit",
        )
        mixed_activation_manifest = json.loads(
            (mixed_activation_baseline / "manifest.json").read_bytes()
        )
        mixed_activation_stages = mixed_activation_manifest["program"][
            "stages"
        ]
        assert [stage["operation"] for stage in mixed_activation_stages] == [
            "sigmoid",
            "elu",
            "softplus",
            "swish",
        ]
        assert [
            stage["dependencies"] for stage in mixed_activation_stages
        ] == [[], [0], [1], [2]]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OPERATION"]
            for stage in mixed_activation_stages
        ] == [33, 35, 37, 38]
        _run_driver(
            driver,
            target,
            mixed_activation_request,
            mixed_activation_baseline,
            "success",
            "valid_mixed_sigmoid_elu_softplus_swish_dag",
        )

        mixed_gelu_document = _mixed_gelu_request(
            identity["identity_sha256"], target
        )
        mixed_gelu_request = fixture_root / "request-mixed-gelu.json"
        mixed_gelu_baseline = fixture_root / "baseline-mixed-gelu"
        _write_request(mixed_gelu_request, mixed_gelu_document)
        provider.compile_request(
            mixed_gelu_request,
            mixed_gelu_baseline,
            "libtriton_jit",
        )
        mixed_gelu_manifest = json.loads(
            (mixed_gelu_baseline / "manifest.json").read_bytes()
        )
        mixed_gelu_stages = mixed_gelu_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_gelu_stages] == [
            "gelu",
            "gelu_approx_tanh",
        ]
        assert [
            stage["dependencies"] for stage in mixed_gelu_stages
        ] == [[], [0]]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OPERATION"]
            for stage in mixed_gelu_stages
        ] == [36, 39]
        _run_driver(
            driver,
            target,
            mixed_gelu_request,
            mixed_gelu_baseline,
            "success",
            "valid_mixed_gelu_gelu_approx_tanh_dag",
        )

        relu_request, relu_baseline, relu_request_document = unary_fixtures[
            "relu"
        ]
        relu_manifest = json.loads(
            (relu_baseline / "manifest.json").read_bytes()
        )

        def write_relu_manifest_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(relu_baseline, artifact)
            changed = copy.deepcopy(relu_manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                relu_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        write_relu_manifest_mutation(
            "unary_platform_source_family_mismatch",
            lambda document: document["program"]["stages"][0][
                "kernel"
            ].__setitem__("source", "binary.py"),
        )
        write_relu_manifest_mutation(
            "unary_relu_meta_mismatch",
            lambda document: _set_stage_meta(
                document, 0, "negative_slope", 0.25
            ),
        )

        def write_unary_request_mutation(
            case_name: str,
            mutate: Callable[[dict[str, Any]], None],
            *,
            base_artifact: Path = relu_baseline,
            base_document: dict[str, object] = relu_request_document,
            base_manifest: dict[str, Any] = relu_manifest,
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(base_artifact, artifact)
            changed_request = copy.deepcopy(base_document)
            mutate(changed_request)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(
                changed_request_path, changed_request
            )
            changed_manifest = copy.deepcopy(base_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (artifact / "manifest.json").write_text(
                json.dumps(
                    changed_manifest,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                artifact,
                "compilation_failed",
                case_name,
            )

        write_unary_request_mutation(
            "unary_relu_raw_normalized_mismatch",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("relu_lower_clip_slope", 0.5),
        )
        write_unary_request_mutation(
            "unary_relu_invalid_clip_interval",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].update(
                {
                    "lower_clip": 4.0,
                    "relu_lower_clip": 4.0,
                }
            ),
        )
        neg_request, neg_baseline, neg_document = unary_fixtures["neg"]
        neg_manifest = json.loads(
            (neg_baseline / "manifest.json").read_bytes()
        )
        write_unary_request_mutation(
            "unary_non_relu_carries_relu_attributes",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].update(
                {
                    "negative_slope": 0.2,
                    "relu_lower_clip_slope": 0.2,
                }
            ),
            base_artifact=neg_baseline,
            base_document=neg_document,
            base_manifest=neg_manifest,
        )
        sqrt_request, sqrt_baseline, sqrt_document = unary_fixtures["sqrt"]
        sqrt_manifest = json.loads(
            (sqrt_baseline / "manifest.json").read_bytes()
        )
        write_unary_request_mutation(
            "unary_sqrt_mode_mismatch",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("mode", UNARY_POINTWISE_MODES["rsqrt"]),
            base_artifact=sqrt_baseline,
            base_document=sqrt_document,
            base_manifest=sqrt_manifest,
        )
        log_request, log_baseline, log_document = unary_fixtures["log"]
        log_manifest = json.loads(
            (log_baseline / "manifest.json").read_bytes()
        )
        write_unary_request_mutation(
            "unary_log_mode_mismatch",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("mode", UNARY_POINTWISE_MODES["exp"]),
            base_artifact=log_baseline,
            base_document=log_document,
            base_manifest=log_manifest,
        )

        sigmoid_request, sigmoid_baseline, sigmoid_document = unary_fixtures[
            "sigmoid"
        ]
        sigmoid_manifest = json.loads(
            (sigmoid_baseline / "manifest.json").read_bytes()
        )
        write_unary_request_mutation(
            "unary_sigmoid_mode_mismatch",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("mode", UNARY_POINTWISE_MODES["tanh"]),
            base_artifact=sigmoid_baseline,
            base_document=sigmoid_document,
            base_manifest=sigmoid_manifest,
        )

        for operation, foreign_mode in (
            ("gelu", UNARY_POINTWISE_MODES["gelu_approx_tanh"]),
            ("gelu_approx_tanh", UNARY_POINTWISE_MODES["gelu"]),
            ("sin", UNARY_POINTWISE_MODES["cos"]),
            ("cos", UNARY_POINTWISE_MODES["tan"]),
            ("tan", UNARY_POINTWISE_MODES["erf"]),
            ("erf", UNARY_POINTWISE_MODES["sin"]),
        ):
            gelu_request, gelu_baseline, gelu_document = unary_fixtures[
                operation
            ]
            gelu_manifest = json.loads(
                (gelu_baseline / "manifest.json").read_bytes()
            )
            write_unary_request_mutation(
                f"unary_{operation}_mode_mismatch",
                lambda document, mode=foreign_mode: document["graph"][
                    "nodes"
                ][0]["attributes"].__setitem__("mode", mode),
                base_artifact=gelu_baseline,
                base_document=gelu_document,
                base_manifest=gelu_manifest,
            )

        for operation, attribute, foreign_operation, invalid_value in (
            ("elu", "elu_alpha", "sigmoid", 2.0),
            ("softplus", "softplus_beta", "elu", 2.0),
            ("swish", "swish_beta", "softplus", 0.5),
        ):
            operation_request, operation_baseline, operation_document = (
                unary_fixtures[operation]
            )
            operation_manifest = json.loads(
                (operation_baseline / "manifest.json").read_bytes()
            )
            write_unary_request_mutation(
                f"unary_{operation}_graph_meta_mismatch",
                lambda document, name=attribute: document["graph"]["nodes"][0][
                    "attributes"
                ].__setitem__(name, 3.0),
                base_artifact=operation_baseline,
                base_document=operation_document,
                base_manifest=operation_manifest,
            )
            foreign_request, foreign_baseline, foreign_document = (
                unary_fixtures[foreign_operation]
            )
            foreign_manifest = json.loads(
                (foreign_baseline / "manifest.json").read_bytes()
            )
            write_unary_request_mutation(
                f"unary_{foreign_operation}_carries_{attribute}",
                lambda document, name=attribute, value=invalid_value: document[
                    "graph"
                ]["nodes"][0]["attributes"].__setitem__(name, value),
                base_artifact=foreign_baseline,
                base_document=foreign_document,
                base_manifest=foreign_manifest,
            )

            alternative_request = fixture_root / (
                f"request-unary-{operation}-identity-variant.json"
            )
            alternative_artifact = fixture_root / (
                f"baseline-unary-{operation}-identity-variant"
            )
            alternative_document = copy.deepcopy(operation_document)
            alternative_document["graph"]["nodes"][0]["attributes"][
                attribute
            ] = 2.0
            _write_request(alternative_request, alternative_document)
            provider.compile_request(
                alternative_request, alternative_artifact, "libtriton_jit"
            )
            alternative_manifest = json.loads(
                (alternative_artifact / "manifest.json").read_bytes()
            )
            assert (
                operation_manifest["program"]["stages"][0]["autotune"][
                    "candidate_identity"
                ]
                != alternative_manifest["program"]["stages"][0]["autotune"][
                    "candidate_identity"
                ]
            )

        softplus_request, softplus_baseline, softplus_document = (
            unary_fixtures["softplus"]
        )
        softplus_manifest = json.loads(
            (softplus_baseline / "manifest.json").read_bytes()
        )
        write_unary_request_mutation(
            "unary_softplus_nonpositive_beta",
            lambda document: document["graph"]["nodes"][0][
                "attributes"
            ].__setitem__("softplus_beta", 0.0),
            base_artifact=softplus_baseline,
            base_document=softplus_document,
            base_manifest=softplus_manifest,
        )

        for operation in ("div", "min", "max", "mod", "pow"):
            binary_request = fixture_root / f"request-{operation}.json"
            binary_baseline = fixture_root / f"baseline-{operation}"
            binary_document = _request(
                identity["identity_sha256"],
                target,
                operations=(operation, operation),
            )
            _write_request(binary_request, binary_document)
            provider.compile_request(
                binary_request, binary_baseline, "libtriton_jit"
            )
            binary_manifest = json.loads(
                (binary_baseline / "manifest.json").read_bytes()
            )
            assert [
                stage["operation"]
                for stage in binary_manifest["program"]["stages"]
            ] == [operation, operation]
            assert [
                stage["candidates"][0]["payload"]["meta"]["OP_KIND"]
                for stage in binary_manifest["program"]["stages"]
            ] == [POINTWISE_MODES[operation], POINTWISE_MODES[operation]]
            _run_driver(
                driver,
                target,
                binary_request,
                binary_baseline,
                "success",
                f"valid_{operation}_dag",
            )

        mixed_mod_pow_document = _request(
            identity["identity_sha256"],
            target,
            autotune=True,
            operations=("mod", "pow"),
        )
        mixed_mod_pow_request = fixture_root / "request-mixed-mod-pow.json"
        mixed_mod_pow_baseline = fixture_root / "baseline-mixed-mod-pow"
        _write_request(mixed_mod_pow_request, mixed_mod_pow_document)
        provider.compile_request(
            mixed_mod_pow_request,
            mixed_mod_pow_baseline,
            "libtriton_jit",
        )
        mixed_mod_pow_manifest = json.loads(
            (mixed_mod_pow_baseline / "manifest.json").read_bytes()
        )
        mixed_mod_pow_stages = mixed_mod_pow_manifest["program"]["stages"]
        assert [stage["operation"] for stage in mixed_mod_pow_stages] == [
            "mod",
            "pow",
        ]
        assert [stage["dependencies"] for stage in mixed_mod_pow_stages] == [
            [],
            [0],
        ]
        assert [
            stage["candidates"][0]["payload"]["meta"]["OP_KIND"]
            for stage in mixed_mod_pow_stages
        ] == [22, 23]
        assert [
            len(stage["argument_sources"]) for stage in mixed_mod_pow_stages
        ] == [4, 4]
        assert [
            [
                candidate["payload"]["meta"]["BLOCK_SIZE"]
                for candidate in stage["candidates"]
            ]
            for stage in mixed_mod_pow_stages
        ] == [[256, 128], [256, 128]]
        assert (
            mixed_mod_pow_stages[0]["autotune"]["candidate_identity"]
            != mixed_mod_pow_stages[1]["autotune"]["candidate_identity"]
        )
        _run_driver(
            driver,
            target,
            mixed_mod_pow_request,
            mixed_mod_pow_baseline,
            "success",
            "valid_mixed_mod_pow_dag",
        )

        for operation in ("mod", "pow"):
            for storage_data_type in ("float16", "bfloat16"):
                typed_request = fixture_root / (
                    f"request-{operation}-{storage_data_type}.json"
                )
                typed_baseline = fixture_root / (
                    f"baseline-{operation}-{storage_data_type}"
                )
                _write_request(
                    typed_request,
                    _request(
                        identity["identity_sha256"],
                        target,
                        storage_data_type=storage_data_type,
                        operations=(operation, operation),
                    ),
                )
                provider.compile_request(
                    typed_request, typed_baseline, "libtriton_jit"
                )
                _run_driver(
                    driver,
                    target,
                    typed_request,
                    typed_baseline,
                    "success",
                    f"valid_{operation}_{storage_data_type}_dag",
                )

        for node_index, foreign_mode in ((0, 23), (1, 22)):
            changed_document = copy.deepcopy(mixed_mod_pow_document)
            changed_document["graph"]["nodes"][node_index]["attributes"][
                "pointwise_mode"
            ] = foreign_mode
            case_name = (
                "graph_mod_mode_mismatch"
                if node_index == 0
                else "graph_pow_mode_mismatch"
            )
            changed_request = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(changed_request, changed_document)
            changed_artifact = fixture_root / case_name
            shutil.copytree(mixed_mod_pow_baseline, changed_artifact)
            changed_manifest = copy.deepcopy(mixed_mod_pow_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (changed_artifact / "manifest.json").write_text(
                json.dumps(
                    changed_manifest,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request,
                changed_artifact,
                "compilation_failed",
                case_name,
            )

        for node_index, operation in ((0, "mod"), (1, "pow")):
            changed_document = copy.deepcopy(mixed_mod_pow_document)
            changed_document["graph"]["nodes"][node_index]["attributes"][
                "alpha"
            ] = 2.0
            case_name = f"graph_{operation}_nondefault_alpha"
            changed_request = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(changed_request, changed_document)
            changed_artifact = fixture_root / case_name
            shutil.copytree(mixed_mod_pow_baseline, changed_artifact)
            changed_manifest = copy.deepcopy(mixed_mod_pow_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (changed_artifact / "manifest.json").write_text(
                json.dumps(
                    changed_manifest,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request,
                changed_artifact,
                "compilation_failed",
                case_name,
            )

        def write_mixed_manifest_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(mixed_baseline, artifact)
            changed = copy.deepcopy(mixed_manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                mixed_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        def mutate_stage_operation(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["operation"] = "add"

        def mutate_stage_op_kind(document: dict[str, Any]) -> None:
            _set_stage_meta(document, 0, "OP_KIND", POINTWISE_MODES["add"])

        def mutate_mul_payload_alpha(document: dict[str, Any]) -> None:
            _set_stage_meta(document, 1, "ALPHA", 2.0)

        def mutate_platform_source(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["kernel"][
                "source"
            ] = "common/binary.py"

        for case_name, manifest_mutator in (
            ("mixed_stage_operation_type_mismatch", mutate_stage_operation),
            ("mixed_stage_op_kind_mismatch", mutate_stage_op_kind),
            ("mixed_mul_payload_alpha_mismatch", mutate_mul_payload_alpha),
            ("mixed_platform_source_mismatch", mutate_platform_source),
        ):
            write_mixed_manifest_mutation(case_name, manifest_mutator)

        def write_mixed_request_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(mixed_baseline, artifact)
            changed_request = copy.deepcopy(mixed_request_document)
            mutate(changed_request)
            changed_request_path = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(
                changed_request_path, changed_request
            )
            changed_manifest = copy.deepcopy(mixed_manifest)
            changed_manifest["request_sha256"] = request_sha256
            (artifact / "manifest.json").write_text(
                json.dumps(
                    changed_manifest,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                changed_request_path,
                artifact,
                "compilation_failed",
                case_name,
            )

        def mutate_sub_graph_mode(document: dict[str, Any]) -> None:
            document["graph"]["nodes"][0]["attributes"]["pointwise_mode"] = (
                POINTWISE_MODES["add"]
            )

        def mutate_mul_graph_alpha(document: dict[str, Any]) -> None:
            document["graph"]["nodes"][1]["attributes"]["alpha"] = 2.0

        def mutate_graph_unsupported_operation(
            document: dict[str, Any]
        ) -> None:
            document["graph"]["nodes"][0]["type"] = "unsupported"
            document["graph"]["nodes"][0]["attributes"]["pointwise_mode"] = 8

        for case_name, request_mutator in (
            ("graph_sub_type_mode_mismatch", mutate_sub_graph_mode),
            ("graph_mul_nondefault_alpha", mutate_mul_graph_alpha),
            (
                "graph_operation_outside_registry_unsupported",
                mutate_graph_unsupported_operation,
            ),
        ):
            write_mixed_request_mutation(case_name, request_mutator)

        autotune_request = fixture_root / "request-autotune.json"
        autotune_baseline = fixture_root / "baseline-autotune"
        _write_request(
            autotune_request,
            _request(identity["identity_sha256"], target, autotune=True),
        )
        provider.compile_request(
            autotune_request, autotune_baseline, "libtriton_jit"
        )
        autotune_manifest = json.loads(
            (autotune_baseline / "manifest.json").read_bytes()
        )
        for stage in autotune_manifest["program"]["stages"]:
            assert len(stage["candidates"]) == 2
            assert stage["autotune"]["enabled"] is True
            assert stage["autotune"]["selection"] == {
                "state": "pending",
                "candidate_id": "",
            }
            assert [
                candidate["payload"]["meta"]["BLOCK_SIZE"]
                for candidate in stage["candidates"]
            ] == [256, 128]
        _run_driver(
            driver,
            target,
            autotune_request,
            autotune_baseline,
            "success",
            "valid_pending_autotune_candidate_set",
        )

        mixed_autotune_request = fixture_root / "request-mixed-autotune.json"
        mixed_autotune_baseline = fixture_root / "baseline-mixed-autotune"
        _write_request(
            mixed_autotune_request,
            _mixed_binary_request(
                identity["identity_sha256"], target, autotune=True
            ),
        )
        provider.compile_request(
            mixed_autotune_request,
            mixed_autotune_baseline,
            "libtriton_jit",
        )
        mixed_autotune_manifest = json.loads(
            (mixed_autotune_baseline / "manifest.json").read_bytes()
        )
        assert [
            stage["operation"]
            for stage in mixed_autotune_manifest["program"]["stages"]
        ] == ["sub", "mul"]
        _run_driver(
            driver,
            target,
            mixed_autotune_request,
            mixed_autotune_baseline,
            "success",
            "valid_mixed_sub_mul_autotune_identities",
        )

        def write_autotune_mutation(
            case_name: str, mutate: Callable[[dict[str, Any]], None]
        ) -> None:
            artifact = fixture_root / case_name
            shutil.copytree(autotune_baseline, artifact)
            changed = copy.deepcopy(autotune_manifest)
            mutate(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                autotune_request,
                artifact,
                "compilation_failed",
                case_name,
            )

        def mutate_candidate_identity(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["autotune"][
                "candidate_identity"
            ] = ("0" * 64)

        def mutate_tuning_source(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["autotune"]["source_sha256"] = (
                "0" * 64
            )

        def mutate_candidate_id(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["candidates"][1][
                "candidate_id"
            ] = ("config_" + "0" * 64)

        def mutate_candidate_option(document: dict[str, Any]) -> None:
            document["program"]["stages"][0]["candidates"][1]["payload"][
                "compile_options"
            ]["num_warps"] = 8

        def mutate_candidate_meta(document: dict[str, Any]) -> None:
            candidate = document["program"]["stages"][0]["candidates"][1]
            candidate["payload"]["meta"]["BLOCK_SIZE"] = 64
            tokens = candidate["payload"]["full_signature"].split(",")
            tokens[-2] = "64"
            candidate["payload"]["full_signature"] = ",".join(tokens)

        def mutate_candidate_order(document: dict[str, Any]) -> None:
            candidates = document["program"]["stages"][0]["candidates"]
            candidates.reverse()

        def mutate_autotune_downgrade(document: dict[str, Any]) -> None:
            stage = document["program"]["stages"][0]
            stage["candidates"] = [stage["candidates"][0]]
            stage["candidates"][0]["candidate_id"] = "default"
            stage["autotune"] = {
                "schema_version": 1,
                "enabled": False,
                "selection": {"state": "fixed", "candidate_id": "default"},
            }

        for case_name, autotune_mutator in (
            ("autotune_candidate_identity", mutate_candidate_identity),
            ("autotune_tuning_source", mutate_tuning_source),
            ("autotune_candidate_id", mutate_candidate_id),
            ("autotune_candidate_option", mutate_candidate_option),
            ("autotune_candidate_meta", mutate_candidate_meta),
            ("autotune_candidate_order", mutate_candidate_order),
            ("autotune_requested_but_downgraded", mutate_autotune_downgrade),
        ):
            write_autotune_mutation(case_name, autotune_mutator)

        boundary_fixtures: dict[int, tuple[str, Path, Path]] = {}
        boundary_identities: set[str] = set()
        for ai_core_count in (1, 24, 65535):
            boundary_target = f"{target_prefix}_aic_{ai_core_count}"
            boundary_identity = provider.compiler_identity(
                boundary_target, "libtriton_jit"
            )
            boundary_identities.add(boundary_identity["identity_sha256"])
            boundary_request = fixture_root / (
                f"request-aic-{ai_core_count}.json"
            )
            boundary_artifact = fixture_root / f"baseline-aic-{ai_core_count}"
            _write_request(
                boundary_request,
                _request(
                    boundary_identity["identity_sha256"], boundary_target
                ),
            )
            provider.compile_request(
                boundary_request, boundary_artifact, "libtriton_jit"
            )
            boundary_manifest = json.loads(
                (boundary_artifact / "manifest.json").read_bytes()
            )
            assert boundary_manifest["target"] == boundary_target
            assert boundary_manifest["compiler"] == boundary_identity
            expected_worker_count = ai_core_count * 2
            for stage in boundary_manifest["program"]["stages"]:
                payload = stage["candidates"][0]["payload"]
                assert payload["meta"]["WORKER_COUNT"] == (
                    expected_worker_count
                )
                assert payload["grid"] == [1, 1, 1]
                assert payload["full_signature"].split(",")[-1] == str(
                    expected_worker_count
                )
            _run_driver(
                driver,
                boundary_target,
                boundary_request,
                boundary_artifact,
                "success",
                f"valid_aic_{ai_core_count}_identity_and_launch",
                ai_core_count=ai_core_count,
            )
            mismatched_ai_core_count = (
                ai_core_count + 1
                if ai_core_count < 65535
                else ai_core_count - 1
            )
            _run_driver(
                driver,
                boundary_target,
                boundary_request,
                boundary_artifact,
                "compilation_failed",
                f"context_aic_{ai_core_count}_target_mismatch",
                ai_core_count=mismatched_ai_core_count,
            )
            boundary_fixtures[ai_core_count] = (
                boundary_target,
                boundary_request,
                boundary_artifact,
            )
        assert len(boundary_identities) == 3

        for supported_soc in SUPPORTED_CODEGEN_ARCHES:
            supported_target = (
                f"ascend_{supported_soc}_cann_{target_cann}_aic_24"
            )
            supported_request = fixture_root / (
                f"request-soc-{supported_soc}.json"
            )
            supported_artifact = fixture_root / (
                f"baseline-soc-{supported_soc}"
            )
            with mock.patch.dict(
                os.environ, {"TRITON_ASCEND_ARCH": supported_soc}
            ):
                supported_identity = provider.compiler_identity(
                    supported_target, "libtriton_jit"
                )
                _write_request(
                    supported_request,
                    _request(
                        supported_identity["identity_sha256"],
                        supported_target,
                    ),
                )
                provider.compile_request(
                    supported_request,
                    supported_artifact,
                    "libtriton_jit",
                )
            supported_manifest = json.loads(
                (supported_artifact / "manifest.json").read_bytes()
            )
            assert supported_manifest["target"] == supported_target
            assert supported_manifest["compiler"] == supported_identity
            _run_driver(
                driver,
                supported_target,
                supported_request,
                supported_artifact,
                "success",
                f"valid_exact_case_soc_{supported_soc}",
                ai_core_count=24,
            )

        one_core_target, one_core_request, one_core_artifact = (
            boundary_fixtures[1]
        )
        assert one_core_target != target
        _run_driver(
            driver,
            target,
            one_core_request,
            one_core_artifact,
            "compilation_failed",
            "context_target_differs_from_request_and_artifact",
            ai_core_count=aicore_count_from_target(target),
        )

        invalid_targets = {
            "wrong_soc_case": (
                f"ascend_{target_soc.lower()}_cann_{target_cann}_aic_24"
            ),
            "zero": f"{target_prefix}_aic_0",
            "overflow": f"{target_prefix}_aic_65536",
            "negative": f"{target_prefix}_aic_-1",
            "leading_zero": f"{target_prefix}_aic_024",
            "missing_marker": f"{target_prefix}_24",
            "duplicate_marker": f"{target_prefix}_aic_24_aic_24",
            "extra_suffix": f"{target_prefix}_aic_24_extra",
        }
        for case_name, invalid_target in invalid_targets.items():
            _run_driver(
                driver,
                invalid_target,
                request_path,
                baseline,
                "compilation_failed",
                f"invalid_target_{case_name}",
                ai_core_count=24,
            )

        large_request_path = fixture_root / "request-large-channels-last.json"
        large_baseline = fixture_root / "baseline-large-channels-last"
        _write_request(
            large_request_path,
            _large_channels_last_dense_request(
                identity["identity_sha256"], target
            ),
        )
        provider.compile_request(
            large_request_path, large_baseline, "libtriton_jit"
        )
        large_manifest = json.loads(
            (large_baseline / "manifest.json").read_bytes()
        )
        assert _payload(large_manifest)["grid"] == [48, 1, 1]
        _run_driver(
            driver,
            target,
            large_request_path,
            large_baseline,
            "success",
            "valid_large_channels_last_dense_grid_cap",
        )
        large_grid_tamper = fixture_root / "large-grid-tamper"
        shutil.copytree(large_baseline, large_grid_tamper)
        _payload(large_manifest)["grid"] = [49, 1, 1]
        (large_grid_tamper / "manifest.json").write_text(
            json.dumps(
                large_manifest, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        _run_driver(
            driver,
            target,
            large_request_path,
            large_grid_tamper,
            "compilation_failed",
            "large_grid_exceeds_worker_cap",
        )

        manifest = json.loads((baseline / "manifest.json").read_bytes())
        for case_name, artifact_mutator in _mutations(target):
            artifact = fixture_root / case_name
            shutil.copytree(baseline, artifact)
            changed = copy.deepcopy(manifest)
            artifact_mutator(changed)
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                request_path,
                artifact,
                "compilation_failed",
                case_name,
            )

        for storage_data_type in ("float16", "bfloat16"):
            _, valid_artifact = valid_fixtures[storage_data_type]
            case_name = f"invalid_{storage_data_type}_compute"
            artifact = fixture_root / case_name
            shutil.copytree(valid_artifact, artifact)
            request_path = fixture_root / f"request-{case_name}.json"
            request_sha256 = _write_request(
                request_path,
                _request(
                    identity["identity_sha256"],
                    target,
                    storage_data_type=storage_data_type,
                    compute_data_type=storage_data_type,
                ),
            )
            changed = json.loads((artifact / "manifest.json").read_bytes())
            changed["request_sha256"] = request_sha256
            (artifact / "manifest.json").write_text(
                json.dumps(changed, sort_keys=True, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            _run_driver(
                driver,
                target,
                request_path,
                artifact,
                "compilation_failed",
                case_name,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
