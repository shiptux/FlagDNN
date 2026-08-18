#!/usr/bin/env python3
"""Fresh-cache offline BiSheng contract for Ascend LayerNorm."""

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
    raise RuntimeError("offline LayerNorm matrix requires assertions enabled")

ROOT = Path(__file__).resolve().parents[3]
STANDALONE_VALUE = os.environ.get("FLAGDNN_STANDALONE_COMPILE")
if not STANDALONE_VALUE:
    raise RuntimeError(
        "FLAGDNN_STANDALONE_COMPILE must name standalone_compile.py"
    )
STANDALONE = Path(STANDALONE_VALUE).expanduser().resolve()
if not STANDALONE.is_file():
    raise RuntimeError(f"standalone compiler was not found: {STANDALONE}")
OUT = Path(tempfile.mkdtemp(prefix="flagdnn-layernorm-offline."))

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
from tools.test_ascend_compiler_contract import TARGET, _compile, _layernorm_request
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

cases: list[tuple[str, dict[str, object], int]] = []
for dtype in ("float32", "float16", "bfloat16"):
    cases.append(
        (
            f"{dtype}-suffix1-autotune",
            _layernorm_request(
                identity["identity_sha256"],
                data_type=dtype,
                dimensions=[2, 5, 17],
                normalized_rank=1,
                autotune=True,
            ),
            2,
        )
    )
    cases.append(
        (
            f"{dtype}-suffix2-autotune",
            _layernorm_request(
                identity["identity_sha256"],
                data_type=dtype,
                dimensions=[2, 3, 4, 5],
                normalized_rank=2,
                autotune=True,
            ),
            2,
        )
    )
cases.extend(
    [
        (
            "float32-rank1-large-autotune",
            _layernorm_request(
                identity["identity_sha256"],
                data_type="float32",
                dimensions=[12289],
                normalized_rank=1,
                autotune=True,
            ),
            2,
        ),
        (
            "float32-suffix1-fixed",
            _layernorm_request(
                identity["identity_sha256"],
                data_type="float32",
                dimensions=[2, 5, 17],
                normalized_rank=1,
                autotune=False,
            ),
            1,
        ),
    ]
)

expected_abi = [
    {"index": 0, "name": "x_ptr", "type": "pointer"},
    {"index": 1, "name": "scale_ptr", "type": "pointer"},
    {"index": 2, "name": "bias_ptr", "type": "pointer"},
    {"index": 3, "name": "y_ptr", "type": "pointer"},
    {"index": 4, "name": "mean_ptr", "type": "pointer"},
    {"index": 5, "name": "inv_variance_ptr", "type": "pointer"},
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
expected_total = sum(candidate_count for _, _, candidate_count in cases)
assert expected_total == 15
for case_name, graph_request, candidate_count in cases:
    print(f"START case={case_name}", flush=True)
    case_root = OUT / "cases" / case_name
    manifest, _ = _compile(provider, graph_request, case_root)
    stages = manifest["program"]["stages"]
    assert len(stages) == 1
    stage = stages[0]
    assert stage["operation"] == "layernorm"
    assert stage["kernel_family"] == "layernorm"
    assert stage["kernel"]["source"] == "normalization.py"
    assert stage["kernel"]["entry_point"] == "layernorm_persistent_kernel"
    assert [item["name"] for item in stage["argument_sources"]] == [
        "x_ptr",
        "scale_ptr",
        "bias_ptr",
        "y_ptr",
        "mean_ptr",
        "inv_variance_ptr",
        "n_elements",
    ]
    graph = graph_request["graph"]
    dimensions = list(graph["tensors"][0]["dimensions"])
    dtype = str(graph["tensors"][0]["data_type"])
    attributes = graph["nodes"][0]["attributes"]
    rows = int(attributes["rows"])
    normalized_elements = int(attributes["normalized_elements"])
    n_elements = math.prod(dimensions)
    assert rows * normalized_elements == n_elements
    source = (
        case_root
        / "artifact"
        / stage["kernel"]["materialized_source"]["file"]
    ).resolve(strict=True)
    assert source.is_file() and not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    candidates = stage["candidates"]
    assert len(candidates) == candidate_count
    expected_blocks = [256, 128] if candidate_count == 2 else [256]
    assert [
        candidate["payload"]["meta"]["BLOCK_SIZE"]
        for candidate in candidates
    ] == expected_blocks
    for candidate in candidates:
        payload = candidate["payload"]
        block = int(payload["meta"]["BLOCK_SIZE"])
        assert payload["entry_point"] == "layernorm_persistent_kernel"
        assert payload["argument_abi"] == expected_abi
        assert payload["meta"] == {
            "ROWS": rows,
            "NORMALIZED_ELEMENTS": normalized_elements,
            "EPSILON": 1.0e-5,
            "BLOCK_SIZE": block,
            "WORKER_COUNT": 48,
        }
        assert payload["grid"] == [min(rows, 48), 1, 1]
        assert payload["compile_options"] == {"num_warps": 4, "num_stages": 1}
        signature = payload["full_signature"].split(",")
        assert signature[:7] == [
            pointer_token[dtype],
            pointer_token[dtype],
            pointer_token[dtype],
            pointer_token[dtype],
            "*fp32:16",
            "*fp32:16",
            "i32",
        ]
        assert len(signature) == 12
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
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
            {"type": "ptr", "dtype": metadata_dtype[dtype]},
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
        assert ttir.count("tt.store") >= 3
        completed += 1
        print(
            f"PASS {completed}/{expected_total} case={case_name} "
            f"block={block} cache={cache}",
            flush=True,
        )

assert completed == expected_total
print(
    f"PASS LayerNorm offline matrix completed={expected_total} out={OUT}",
    flush=True,
)
