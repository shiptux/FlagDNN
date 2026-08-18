#!/usr/bin/env python3
"""Fresh-cache offline BiSheng contract for Ascend BatchNorm training."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import re
import signal
import sys
import tempfile
import time
import traceback


if not __debug__:
    raise RuntimeError("offline BatchNorm training matrix requires assertions")

ROOT = Path(__file__).resolve().parents[3]
STANDALONE_VALUE = os.environ.get("FLAGDNN_STANDALONE_COMPILE")
if not STANDALONE_VALUE:
    raise RuntimeError(
        "FLAGDNN_STANDALONE_COMPILE must name standalone_compile.py"
    )
STANDALONE = Path(STANDALONE_VALUE).expanduser().resolve()
if not STANDALONE.is_file():
    raise RuntimeError(f"standalone compiler was not found: {STANDALONE}")
OUT = Path(tempfile.mkdtemp(prefix="flagdnn-batchnorm-training-offline."))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compiler"))
for cache_variable in (
    "TRITON_CACHE_DIR",
    "TRITON_OVERRIDE_DIR",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_REMOTE_CACHE_BACKEND",
    "TRITON_CACHE_MANAGER",
):
    os.environ.pop(cache_variable, None)
os.environ["TRITON_JIT_BACKEND"] = "NPU"
os.environ["TRITON_ASCEND_ARCH"] = "Ascend910B1"
os.environ["TRITON_CACHE_DIR"] = str(OUT / "cache")

from flagdnn_codegen.provider_loader import get_provider
from tools.test_ascend_compiler_contract import TARGET, _batchnorm_request, _compile
from triton.backends.ascend.driver import NPUUtils


NPUUtils.get_arch = lambda _self: os.environ["TRITON_ASCEND_ARCH"]

spec = importlib.util.spec_from_file_location(
    "flagdnn_exact_standalone", STANDALONE
)
assert spec is not None and spec.loader is not None
standalone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standalone)


def compile_worker(result_queue: object, arguments: tuple[object, ...]) -> None:
    os.setsid()
    result_queue.put(("ready", ""))
    try:
        cache = standalone.compile_a_kernel(*arguments)
        result_queue.put(("ok", str(cache)))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def compile_with_hard_timeout(
    arguments: tuple[object, ...], timeout_seconds: float = 120
) -> Path:
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    worker = context.Process(target=compile_worker, args=(result_queue, arguments))
    started = False
    deadline = time.monotonic() + timeout_seconds
    try:
        worker.start()
        started = True
        status, result = result_queue.get(
            timeout=min(5, max(0.001, deadline - time.monotonic()))
        )
        if status != "ready":
            raise RuntimeError(f"offline BiSheng worker was not ready: {result}")
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            raise TimeoutError("offline BiSheng candidate exceeded 120 seconds")
        status, result = result_queue.get(timeout=5)
        if worker.exitcode != 0 or status != "ok":
            raise RuntimeError(f"offline BiSheng worker failed: {result}")
        return Path(result)
    except queue.Empty as error:
        raise RuntimeError(
            f"offline BiSheng worker exited {worker.exitcode} without result"
        ) from error
    finally:
        if started and worker.is_alive():
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except OSError:
                try:
                    worker.kill()
                except OSError:
                    pass
        try:
            if started:
                worker.join()
        finally:
            try:
                result_queue.close()
            finally:
                result_queue.join_thread()


provider = get_provider("ascend")
identity = provider.compiler_identity(TARGET, "libtriton_jit")
source_sha = hashlib.sha256(
    (ROOT / "backends/ascend/kernels/normalization.py").read_bytes()
).hexdigest()

dimensions = [2, 3, 2, 2]
layouts = {
    "contiguous": ([12, 4, 2, 1], [12, 4, 2, 1]),
    "channels-last-gapped": ([30, 1, 12, 4], [40, 1, 16, 5]),
}
cases: list[tuple[str, dict[str, object], int]] = []
for dtype in ("float32", "float16", "bfloat16"):
    for layout, (x_strides, y_strides) in layouts.items():
        for autotune in (False, True):
            cases.append(
                (
                    f"{dtype}-{layout}-{'autotune' if autotune else 'fixed'}",
                    _batchnorm_request(
                        identity["identity_sha256"],
                        data_type=dtype,
                        dimensions=dimensions,
                        x_strides=x_strides,
                        y_strides=y_strides,
                        autotune=autotune,
                    ),
                    2 if autotune else 1,
                )
            )

expected_abi = [
    {"index": 0, "name": "x_ptr", "type": "pointer"},
    {"index": 1, "name": "scale_ptr", "type": "pointer"},
    {"index": 2, "name": "bias_ptr", "type": "pointer"},
    {"index": 3, "name": "previous_running_mean_ptr", "type": "pointer"},
    {"index": 4, "name": "previous_running_variance_ptr", "type": "pointer"},
    {"index": 5, "name": "y_ptr", "type": "pointer"},
    {"index": 6, "name": "mean_ptr", "type": "pointer"},
    {"index": 7, "name": "inv_variance_ptr", "type": "pointer"},
    {"index": 8, "name": "next_running_mean_ptr", "type": "pointer"},
    {"index": 9, "name": "next_running_variance_ptr", "type": "pointer"},
    {"index": 10, "name": "n_elements", "type": "i32"},
]
pointer_token = {
    "float32": "*fp32:16",
    "float16": "*fp16:16",
    "bfloat16": "*bf16:16",
}
metadata_dtype = {
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}

completed = 0
expected_total = sum(candidate_count for _, _, candidate_count in cases)
assert expected_total == 18
for case_name, graph_request, candidate_count in cases:
    print(f"START case={case_name}", flush=True)
    case_root = OUT / "cases" / case_name
    manifest, _ = _compile(provider, graph_request, case_root)
    stages = manifest["program"]["stages"]
    assert len(stages) == 1
    stage = stages[0]
    assert stage["operation"] == "batchnorm"
    assert stage["kernel_family"] == "batchnorm"
    assert stage["kernel"]["source"] == "normalization.py"
    assert stage["kernel"]["entry_point"] == "batchnorm_training_persistent_kernel"
    assert [argument["name"] for argument in stage["argument_sources"]] == [
        argument["name"] for argument in expected_abi
    ]

    graph = graph_request["graph"]
    x = graph["tensors"][0]
    y = graph["tensors"][5]
    dtype = str(x["data_type"])
    x_strides = list(x["strides"])
    y_strides = list(y["strides"])
    attributes = graph["nodes"][0]["attributes"]
    rank = len(dimensions)
    batch = dimensions[0]
    channels = dimensions[1]
    spatial = math.prod(dimensions[2:])
    reduction_elements = batch * spatial
    n_elements = math.prod(dimensions)

    source = (
        case_root / "artifact" / stage["kernel"]["materialized_source"]["file"]
    ).resolve(strict=True)
    assert source.is_file() and not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    candidates = stage["candidates"]
    assert len(candidates) == candidate_count
    expected_blocks = [256, 128] if candidate_count == 2 else [256]
    assert [candidate["payload"]["meta"]["BLOCK_SIZE"] for candidate in candidates] == expected_blocks

    padded_dimensions = [1] * (8 - rank) + dimensions
    padded_x_strides = [0] * (8 - rank) + x_strides
    padded_y_strides = [0] * (8 - rank) + y_strides
    for candidate in candidates:
        payload = candidate["payload"]
        block = int(payload["meta"]["BLOCK_SIZE"])
        assert payload["entry_point"] == "batchnorm_training_persistent_kernel"
        assert payload["argument_abi"] == expected_abi
        expected_meta = {
            "RANK": rank,
            "BATCH": batch,
            "CHANNELS": channels,
            "SPATIAL": spatial,
            "REDUCTION_ELEMENTS": reduction_elements,
            "EPSILON": attributes["epsilon"],
            "MOMENTUM": attributes["momentum"],
            **{f"DIM_{axis}": value for axis, value in enumerate(padded_dimensions)},
            **{f"X_STRIDE_{axis}": value for axis, value in enumerate(padded_x_strides)},
            **{f"Y_STRIDE_{axis}": value for axis, value in enumerate(padded_y_strides)},
            "BLOCK_SIZE": block,
            "WORKER_COUNT": 48,
        }
        assert payload["meta"] == expected_meta
        assert payload["grid"] == [min(channels, 48), 1, 1]
        assert payload["compile_options"] == {"num_warps": 4, "num_stages": 1}
        signature = payload["full_signature"].split(",")
        assert signature[:11] == [
            pointer_token[dtype],
            pointer_token[dtype],
            pointer_token[dtype],
            "*fp32:16",
            "*fp32:16",
            pointer_token[dtype],
            "*fp32:16",
            "*fp32:16",
            "*fp32:16",
            "*fp32:16",
            "i32",
        ]
        assert len(signature) == 44
        cache = compile_with_hard_timeout(
            (
                source,
                payload["entry_point"],
                payload["full_signature"],
                4,
                1,
                0,
            )
        )
        files = [path for path in cache.rglob("*") if path.is_file()]
        assert any(path.suffix == ".npubin" and path.stat().st_size > 0 for path in files)
        metadata_files = [
            path
            for path in files
            if path.suffix == ".json" and not path.name.startswith("__grp__")
        ]
        assert len(metadata_files) == 1
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        assert metadata.get("workspace_size", 0) == 0
        assert metadata["arg_layout"] == [
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": "fp32"},
            {"type": "ptr", "dtype": "fp32"},
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": "fp32"},
            {"type": "ptr", "dtype": "fp32"},
            {"type": "ptr", "dtype": "fp32"},
            {"type": "ptr", "dtype": "fp32"},
            {"type": "i32"},
        ]
        ttir_files = [path for path in files if path.suffix == ".ttir"]
        assert ttir_files
        ttir = ttir_files[0].read_text(encoding="utf-8", errors="replace")
        assert re.search(
            rf"tt\.make_range \{{end = {block} : i32, start = 0 : i32\}}"
            rf" : tensor<{block}xi32>",
            ttir,
        )
        assert f"tensor<{block}xi64>" in ttir
        assert ttir.count("scf.while") >= 4
        assert ttir.count("tt.reduce") >= 2
        assert ttir.count("tt.store") >= 5
        assert n_elements == int(stage["argument_sources"][-1]["value"])
        completed += 1
        print(
            f"PASS {completed}/{expected_total} case={case_name} "
            f"block={block} cache={cache}",
            flush=True,
        )

assert completed == expected_total
print(
    f"PASS BatchNorm training offline matrix completed={expected_total} out={OUT}",
    flush=True,
)
