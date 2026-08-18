#!/usr/bin/env python3
"""Fresh-cache offline BiSheng contract for Ascend convolution FProp."""

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
    raise RuntimeError("offline convolution FProp matrix requires assertions")

ROOT = Path(__file__).resolve().parents[3]
STANDALONE_VALUE = os.environ.get("FLAGDNN_STANDALONE_COMPILE")
if not STANDALONE_VALUE:
    raise RuntimeError(
        "FLAGDNN_STANDALONE_COMPILE must name standalone_compile.py"
    )
STANDALONE = Path(STANDALONE_VALUE).expanduser().resolve()
if not STANDALONE.is_file():
    raise RuntimeError(f"standalone compiler was not found: {STANDALONE}")
OUT = Path(tempfile.mkdtemp(prefix="flagdnn-convolution-fprop-offline."))

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
    _compile,
    _convolution_fprop_request,
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


provider = get_provider("ascend")
identity = provider.compiler_identity(TARGET, "libtriton_jit")
source_sha = hashlib.sha256(
    (ROOT / "backends/ascend/kernels/convolution.py").read_bytes()
).hexdigest()

cases = [
    (
        "rank2-fp16-contiguous-tiled",
        _convolution_fprop_request(
            identity["identity_sha256"],
            data_type="float16",
            input_dimensions=[1, 16, 16, 16],
            filter_dimensions=[32, 16, 3, 3],
            autotune=True,
        ),
    ),
    (
        "rank2-fp32-contiguous-tiled",
        _convolution_fprop_request(
            identity["identity_sha256"],
            input_dimensions=[1, 8, 8, 8],
            filter_dimensions=[16, 8, 1, 1],
            pre_padding=[0, 0],
            post_padding=[0, 0],
            autotune=True,
        ),
    ),
    (
        "rank1-fp16-channels-last",
        _convolution_fprop_request(
            identity["identity_sha256"],
            data_type="float16",
            input_dimensions=[2, 4, 16],
            filter_dimensions=[6, 4, 3],
            output_strides=[96, 1, 6],
            autotune=True,
        ),
    ),
    (
        "rank2-fp32-grouped-asymmetric-gapped",
        _convolution_fprop_request(
            identity["identity_sha256"],
            input_dimensions=[1, 4, 7, 9],
            filter_dimensions=[6, 2, 3, 2],
            input_strides=[500, 100, 12, 1],
            filter_strides=[100, 30, 8, 1],
            output_strides=[400, 60, 10, 1],
            pre_padding=[1, 0],
            post_padding=[2, 1],
            stride=[2, 1],
            dilation=[1, 2],
            groups=2,
            autotune=True,
        ),
    ),
    (
        "rank3-bf16-channels-last",
        _convolution_fprop_request(
            identity["identity_sha256"],
            data_type="bfloat16",
            input_dimensions=[1, 2, 5, 6, 7],
            filter_dimensions=[4, 2, 3, 3, 3],
            input_strides=[420, 1, 84, 14, 2],
            filter_strides=[54, 1, 18, 6, 2],
            output_strides=[840, 1, 168, 28, 4],
            autotune=True,
        ),
    ),
]

expected_abi = [
    {"index": 0, "name": "input_ptr", "type": "pointer"},
    {"index": 1, "name": "filter_ptr", "type": "pointer"},
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
entry_point = "convolution_fprop_persistent_kernel"

completed = 0
expected_total = len(cases) * 2
assert expected_total == 10
for case_name, graph_request in cases:
    print(f"START case={case_name}", flush=True)
    case_root = OUT / "cases" / case_name
    manifest, _ = _compile(provider, graph_request, case_root)
    stage = manifest["program"]["stages"][0]
    dtype = str(graph_request["graph"]["tensors"][0]["data_type"])
    assert stage["operation"] == "convolution_fprop"
    assert stage["kernel_family"] == "convolution_fprop"
    assert stage["kernel"]["source"] == "convolution.py"
    assert stage["kernel"]["entry_point"] == entry_point
    assert stage["argument_sources"][-1] == {
        "index": 3,
        "name": "n_elements",
        "source": "scalar",
        "type": "i32",
        "value": graph_request["graph"]["nodes"][0]["attributes"]["n_outputs"],
    }

    source = (
        case_root / "artifact" / stage["kernel"]["materialized_source"]["file"]
    ).resolve(strict=True)
    assert source.is_file() and not source.is_symlink()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    candidates = stage["candidates"]
    assert len(candidates) == 2
    assert {
        int(candidate["payload"]["meta"]["BLOCK_SIZE"])
        for candidate in candidates
    } == {128, 256}

    for candidate in candidates:
        payload = candidate["payload"]
        meta = payload["meta"]
        block = int(meta["BLOCK_SIZE"])
        assert payload["entry_point"] == entry_point
        assert payload["argument_abi"] == expected_abi
        assert payload["compile_options"] == {"num_warps": 4, "num_stages": 1}
        assert payload["grid"][1:] == [1, 1]
        assert int(payload["grid"][0]) > 0
        signature = payload["full_signature"].split(",")
        assert signature[:4] == [pointer_token[dtype]] * 3 + ["i32"]
        assert len(signature) == 53
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
        if case_name.endswith("-tiled"):
            assert re.search(
                r"tt\.make_range \{end = 8 : i32, start = 0 : i32\}",
                ttir,
            )
            if block == 128:
                assert re.search(
                    r"tt\.make_range \{end = 4 : i32, start = 0 : i32\}",
                    ttir,
                )
        else:
            assert re.search(
                rf"tt\.make_range \{{end = {block} : i32, start = 0 : i32\}}",
                ttir,
            )
        assert "tt.load" in ttir
        assert "tt.store" in ttir
        completed += 1
        print(
            f"PASS {completed}/{expected_total} case={case_name} "
            f"block={block} cache={cache}",
            flush=True,
        )

assert completed == expected_total
print(
    f"PASS convolution FProp offline matrix completed={expected_total} out={OUT}",
    flush=True,
)
