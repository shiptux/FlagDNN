#!/usr/bin/env python3
"""Fresh-cache offline BiSheng contract for Ascend MatMul."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    raise RuntimeError("offline MatMul matrix requires assertions")

ROOT = Path(__file__).resolve().parents[3]
STANDALONE_VALUE = os.environ.get("FLAGDNN_STANDALONE_COMPILE")
if not STANDALONE_VALUE:
    raise RuntimeError(
        "FLAGDNN_STANDALONE_COMPILE must name standalone_compile.py"
    )
STANDALONE = Path(STANDALONE_VALUE).expanduser().resolve()
if not STANDALONE.is_file():
    raise RuntimeError(f"standalone compiler was not found: {STANDALONE}")
OUT = Path(tempfile.mkdtemp(prefix="flagdnn-matmul-offline."))

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
from tools.test_ascend_compiler_contract import TARGET, _compile, _matmul_request
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
    (ROOT / "backends/ascend/kernels/matmul.py").read_bytes()
).hexdigest()

cases: list[tuple[str, dict[str, object], int]] = []
for dtype in ("float32", "float16", "bfloat16"):
    for autotune in (False, True):
        cases.append(
            (
                f"{dtype}-{'autotune' if autotune else 'fixed'}",
                _matmul_request(
                    identity["identity_sha256"],
                    data_type=dtype,
                    a_dimensions=[2, 17, 30],
                    b_dimensions=[2, 30, 23],
                    autotune=autotune,
                ),
                2 if autotune else 1,
            )
        )
cases.append(
    (
        "bfloat16-broadcast-gapped-fixed",
        _matmul_request(
            identity["identity_sha256"],
            data_type="bfloat16",
            a_dimensions=[2, 1, 3, 4],
            b_dimensions=[3, 4, 5],
            a_strides=[100, 100, 5, 1],
            b_strides=[100, 6, 1],
            output_strides=[500, 100, 20, 1],
        ),
        1,
    )
)

expected_abi = [
    {"index": 0, "name": "a_ptr", "type": "pointer"},
    {"index": 1, "name": "b_ptr", "type": "pointer"},
    {"index": 2, "name": "output_ptr", "type": "pointer"},
    {"index": 3, "name": "n_elements", "type": "i32"},
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


def padded_batch_strides(
    dimensions: list[int],
    strides: list[int],
    output_batch_dimensions: list[int],
) -> list[int]:
    batch_dimensions = dimensions[:-2]
    batch_strides = strides[:-2]
    aligned_leading = len(output_batch_dimensions) - len(batch_dimensions)
    aligned_dimensions = [1] * aligned_leading + batch_dimensions
    aligned_strides = [0] * aligned_leading + batch_strides
    return [0] * (6 - len(output_batch_dimensions)) + [
        0 if dimension == 1 else stride
        for dimension, stride in zip(
            aligned_dimensions, aligned_strides, strict=True
        )
    ]


completed = 0
expected_total = sum(candidate_count for _, _, candidate_count in cases)
assert expected_total == 10
for case_name, graph_request, candidate_count in cases:
    print(f"START case={case_name}", flush=True)
    case_root = OUT / "cases" / case_name
    manifest, _ = _compile(provider, graph_request, case_root)
    stage = manifest["program"]["stages"][0]
    assert stage["operation"] == "matmul"
    assert stage["kernel_family"] == "matmul"
    assert stage["kernel"]["source"] == "matmul.py"
    assert stage["kernel"]["entry_point"] == "matmul_strided_kernel"
    assert [argument["name"] for argument in stage["argument_sources"]] == [
        argument["name"] for argument in expected_abi
    ]

    graph = graph_request["graph"]
    a, b, output = graph["tensors"]
    attributes = graph["nodes"][0]["attributes"]
    dtype = str(a["data_type"])
    output_batch_dimensions = list(output["dimensions"][:-2])
    padded_dimensions = [1] * (6 - len(output_batch_dimensions)) + output_batch_dimensions
    a_batch_strides = padded_batch_strides(
        list(a["dimensions"]), list(a["strides"]), output_batch_dimensions
    )
    b_batch_strides = padded_batch_strides(
        list(b["dimensions"]), list(b["strides"]), output_batch_dimensions
    )
    c_batch_strides = [0] * (6 - len(output_batch_dimensions)) + list(
        output["strides"][:-2]
    )

    source = (
        case_root / "artifact" / stage["kernel"]["materialized_source"]["file"]
    ).resolve(strict=True)
    assert source.is_file() and not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    candidates = stage["candidates"]
    assert len(candidates) == candidate_count
    expected_blocks = [16, 32] if candidate_count == 2 else [16]
    assert [candidate["payload"]["meta"]["BLOCK_SIZE"] for candidate in candidates] == expected_blocks

    for candidate in candidates:
        payload = candidate["payload"]
        meta = payload["meta"]
        block = int(meta["BLOCK_SIZE"])
        assert payload["entry_point"] == "matmul_strided_kernel"
        assert payload["argument_abi"] == expected_abi
        assert meta["BATCH"] == attributes["batch"]
        assert meta["M"] == attributes["m"]
        assert meta["N"] == attributes["n"]
        assert meta["K"] == attributes["k"]
        assert [meta[f"DIM_{axis}"] for axis in range(6)] == padded_dimensions
        assert [meta[f"A_BATCH_STRIDE_{axis}"] for axis in range(6)] == a_batch_strides
        assert [meta[f"B_BATCH_STRIDE_{axis}"] for axis in range(6)] == b_batch_strides
        assert [meta[f"C_BATCH_STRIDE_{axis}"] for axis in range(6)] == c_batch_strides
        tiles = ((attributes["m"] + block - 1) // block) * (
            (attributes["n"] + block - 1) // block
        )
        assert payload["grid"] == [
            min(tiles * attributes["batch"], meta["WORKER_COUNT"]),
            1,
            1,
        ]
        assert payload["compile_options"] == {"num_warps": 4, "num_stages": 1}
        signature = payload["full_signature"].split(",")
        assert signature[:4] == [pointer_token[dtype]] * 3 + ["i32"]
        assert len(signature) == 42
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
            {"type": "i32"},
        ]
        ttir_files = [path for path in files if path.suffix == ".ttir"]
        assert ttir_files
        ttir = ttir_files[0].read_text(encoding="utf-8", errors="replace")
        assert re.search(
            rf"tt\.make_range \{{end = {block} : i32, start = 0 : i32\}}",
            ttir,
        )
        assert "tt.store" in ttir
        completed += 1
        print(
            f"PASS {completed}/{expected_total} case={case_name} "
            f"block={block} cache={cache}",
            flush=True,
        )

assert completed == expected_total
print(f"PASS MatMul offline matrix completed={expected_total} out={OUT}", flush=True)
