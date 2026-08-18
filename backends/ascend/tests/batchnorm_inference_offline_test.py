#!/usr/bin/env python3
"""Fresh-cache offline BiSheng contract for Ascend batchnorm inference."""

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
    raise RuntimeError("offline batchnorm matrix requires assertions enabled")

ROOT = Path(__file__).resolve().parents[3]
STANDALONE_VALUE = os.environ.get("FLAGDNN_STANDALONE_COMPILE")
if not STANDALONE_VALUE:
    raise RuntimeError(
        "FLAGDNN_STANDALONE_COMPILE must name standalone_compile.py"
    )
STANDALONE = Path(STANDALONE_VALUE).expanduser().resolve()
if not STANDALONE.is_file():
    raise RuntimeError(f"standalone compiler was not found: {STANDALONE}")
OUT = Path(tempfile.mkdtemp(prefix="flagdnn-batchnorm-offline."))

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
from tools.test_ascend_compiler_contract import (
    TARGET,
    _batchnorm_inference_request,
    _compile,
)
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


def contiguous_strides(dimensions: list[int]) -> list[int]:
    result = [1] * len(dimensions)
    for index in range(len(dimensions) - 2, -1, -1):
        result[index] = result[index + 1] * dimensions[index + 1]
    return result


provider = get_provider("ascend")
identity = provider.compiler_identity(TARGET, "libtriton_jit")
source_sha = hashlib.sha256(
    (ROOT / "backends/ascend/kernels/normalization.py").read_bytes()
).hexdigest()

rank2_dimensions = [2, 5]
rank5_dimensions = [2, 3, 2, 2, 2]
rank8_dimensions = [2, 5, 2, 2, 2, 2, 2, 49]
cases: list[tuple[str, dict[str, object], bool]] = []
for dtype in ("float32", "float16", "bfloat16"):
    cases.append(
        (
            f"{dtype}-rank2-fast",
            _batchnorm_inference_request(
                identity["identity_sha256"],
                data_type=dtype,
                dimensions=rank2_dimensions,
                x_strides=[5, 1],
                y_strides=[5, 1],
                autotune=True,
            ),
            False,
        )
    )
    cases.append(
        (
            f"{dtype}-rank5-generic",
            _batchnorm_inference_request(
                identity["identity_sha256"],
                data_type=dtype,
                dimensions=rank5_dimensions,
                x_strides=[50, 15, 7, 3, 1],
                y_strides=[60, 18, 8, 3, 1],
                autotune=True,
            ),
            True,
        )
    )
cases.append(
    (
        "float32-rank8-persistent-loop",
        _batchnorm_inference_request(
            identity["identity_sha256"],
            data_type="float32",
            dimensions=rank8_dimensions,
            x_strides=contiguous_strides(rank8_dimensions),
            y_strides=contiguous_strides(rank8_dimensions),
            autotune=True,
        ),
        False,
    )
)

expected_abi = [
    {"index": 0, "name": "x_ptr", "type": "pointer"},
    {"index": 1, "name": "mean_ptr", "type": "pointer"},
    {"index": 2, "name": "inv_variance_ptr", "type": "pointer"},
    {"index": 3, "name": "scale_ptr", "type": "pointer"},
    {"index": 4, "name": "bias_ptr", "type": "pointer"},
    {"index": 5, "name": "y_ptr", "type": "pointer"},
    {"index": 6, "name": "n_elements", "type": "i32"},
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
for case_name, graph_request, generic in cases:
    print(f"START case={case_name}", flush=True)
    case_root = OUT / "cases" / case_name
    manifest, _ = _compile(provider, graph_request, case_root)
    stages = manifest["program"]["stages"]
    assert len(stages) == 1
    stage = stages[0]
    assert stage["operation"] == "batchnorm_inference"
    assert stage["kernel_family"] == "batchnorm_inference"
    assert stage["kernel"]["source"] == "normalization.py"
    graph = graph_request["graph"]
    dimensions = list(graph["tensors"][0]["dimensions"])
    x_strides = list(graph["tensors"][0]["strides"])
    y_strides = list(graph["tensors"][5]["strides"])
    dtype = str(graph["tensors"][0]["data_type"])
    n_elements = math.prod(dimensions)
    rank = len(dimensions)
    channels = dimensions[1]
    spatial = math.prod(dimensions[2:])
    source = (
        case_root
        / "artifact"
        / stage["kernel"]["materialized_source"]["file"]
    ).resolve(strict=True)
    assert source.is_file() and not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    candidates = stage["candidates"]
    assert len(candidates) == 2
    assert {candidate["payload"]["meta"]["BLOCK_SIZE"] for candidate in candidates} == {
        128,
        256,
    }
    for candidate in candidates:
        payload = candidate["payload"]
        block = payload["meta"]["BLOCK_SIZE"]
        expected_entry = (
            "batchnorm_inference_strided_persistent_kernel"
            if generic
            else "batchnorm_inference_nchw_persistent_kernel"
        )
        assert payload["entry_point"] == expected_entry
        assert payload["argument_abi"] == expected_abi
        signature = payload["full_signature"].split(",")
        assert signature[:7] == [
            pointer_token[dtype],
            "*fp32:16",
            "*fp32:16",
            "*fp32:16",
            "*fp32:16",
            pointer_token[dtype],
            "i32",
        ]
        assert len(signature) == 7 + len(payload["meta"])
        expected_meta = {
            "RANK": rank,
            "CHANNELS": channels,
            "SPATIAL": spatial,
            "BLOCK_SIZE": block,
            "WORKER_COUNT": 48,
        }
        if generic:
            padded_dimensions = [1] * (8 - rank) + dimensions
            padded_x_strides = [0] * (8 - rank) + x_strides
            padded_y_strides = [0] * (8 - rank) + y_strides
            expected_meta.update(
                {f"DIM_{index}": value for index, value in enumerate(padded_dimensions)}
            )
            expected_meta.update(
                {
                    f"X_STRIDE_{index}": value
                    for index, value in enumerate(padded_x_strides)
                }
            )
            expected_meta.update(
                {
                    f"Y_STRIDE_{index}": value
                    for index, value in enumerate(padded_y_strides)
                }
            )
        assert payload["meta"] == expected_meta
        assert payload["grid"] == [
            min((n_elements + block - 1) // block, 48),
            1,
            1,
        ]
        assert payload["compile_options"] == {"num_warps": 4, "num_stages": 1}
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
        assert any(
            path.suffix == ".npubin" and path.stat().st_size > 0 for path in files
        )
        metadata_files = [
            path
            for path in files
            if path.suffix == ".json" and not path.name.startswith("__grp__")
        ]
        assert len(metadata_files) == 1
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        assert metadata.get("workspace_size", 0) == 0
        assert metadata["arg_layout"] == [
            {"type": "ptr", "dtype": metadata_dtype[dtype]}
            if index in (0, 5)
            else {"type": "ptr", "dtype": "fp32"}
            for index in range(6)
        ] + [{"type": "i32"}]
        ttir_files = [path for path in files if path.suffix == ".ttir"]
        assert ttir_files
        ttir = ttir_files[0].read_text(encoding="utf-8", errors="replace")
        assert re.search(
            rf"tt\.make_range \{{end = {block} : i32, start = 0 : i32\}}"
            rf" : tensor<{block}xi32>",
            ttir,
        )
        assert f"tensor<{block}xi64>" in ttir
        assert "scf.while" in ttir
        assert (
            rf"dense<48> : tensor<{block}xi64>" not in ttir
        ), "lanes must remain contiguous within each persistent block"
        step_constant = re.search(
            rf"(?P<step_constant>%[A-Za-z0-9_]+) = arith\.constant "
            rf"{block * 48} : i64",
            ttir,
        )
        assert step_constant is not None
        step = re.search(
            rf"(?P<next>%[A-Za-z0-9_]+) = arith\.addi [^,\n]+, "
            rf"{re.escape(step_constant.group('step_constant'))} : i64",
            ttir,
        )
        assert step is not None
        assert re.search(
            rf"scf\.yield {re.escape(step.group('next'))} : i64", ttir
        )
        if generic:
            assert "arith.remsi" in ttir
            assert "arith.divsi" in ttir
            assert ttir.count("arith.muli") >= 3
        completed += 1
        print(
            f"PASS {completed}/14 case={case_name} block={block} cache={cache}",
            flush=True,
        )

assert completed == 14
print(f"PASS batchnorm offline matrix completed=14 out={OUT}", flush=True)
