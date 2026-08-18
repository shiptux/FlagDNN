#!/usr/bin/env python3
"""Populate and describe FlagDNN's controlled Ascend Triton cache.

This helper is intentionally a small, schema-versioned wrapper around the
installed libtriton_jit ``standalone_compile.py``. Process containment,
start-gating, quotas, timeouts and journal rollback are owned by the native
launcher; this program never creates or chooses a cache root on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 1 << 20
MAX_TREE_FILES = 4096
MAX_TREE_BYTES = 1 << 30
MAX_RUNTIME_ARGUMENTS = 4096
ALLOWED_RUNTIME_TYPES = {"pointer", "i32", "i64", "f32", "f64"}
DRIVER_BOOTSTRAP_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "mode",
        "cache_root",
        "tmp_root",
        "standalone_compile",
        "standalone_compile_sha256",
        "codegen_environment_digest",
    }
)
COMPILE_CANDIDATE_REQUEST_KEYS = DRIVER_BOOTSTRAP_REQUEST_KEYS | frozenset(
    {
        "source_path",
        "source_sha256",
        "entry_point",
        "full_signature",
        "num_warps",
        "num_stages",
        "device_ordinal",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _open_readonly_no_follow(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("this platform has no O_NOFOLLOW support")
    return os.open(path, flags | no_follow)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = _open_readonly_no_follow(path)
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError(f"cannot hash a non-regular file: {path}")
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(encoded: bytes) -> Any:
    return json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )


def _required_string(document: dict[str, Any], name: str) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _required_sha256(document: dict[str, Any], name: str) -> str:
    value = _required_string(document, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_existing_path(value: str, description: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{description} must be absolute")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{description} must already be canonical")
    return resolved


def _canonical_output_path(path: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("attestation output must be absolute")
    parent = path.parent
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != parent:
        raise ValueError("attestation output parent must already be canonical")
    if not parent.is_dir():
        raise ValueError("attestation output parent must be a directory")
    try:
        descriptor = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return path
    if stat.S_ISLNK(descriptor.st_mode):
        raise ValueError("attestation output must not be a symlink")
    if not stat.S_ISREG(descriptor.st_mode):
        raise ValueError("attestation output must be a regular file when it exists")
    return path


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_request(path: Path, expected_mode: str) -> dict[str, Any]:
    file_descriptor = _open_readonly_no_follow(path)
    with os.fdopen(file_descriptor, "rb") as source:
        descriptor = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(descriptor.st_mode)
            or descriptor.st_size > MAX_REQUEST_BYTES
        ):
            raise ValueError("attestation request is not a bounded regular file")
        encoded = source.read(MAX_REQUEST_BYTES + 1)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("attestation request is not a bounded regular file")
    document = _parse_json(encoded)
    if not isinstance(document, dict):
        raise ValueError("attestation request must be an object")
    schema_version = document.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError("attestation request schema is unsupported")
    if document.get("mode") != expected_mode:
        raise ValueError("attestation request mode differs from CLI mode")
    expected_keys = (
        DRIVER_BOOTSTRAP_REQUEST_KEYS
        if expected_mode == "driver-bootstrap"
        else COMPILE_CANDIDATE_REQUEST_KEYS
    )
    actual_keys = frozenset(document)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "attestation request schema keys differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return document


def _validate_environment(document: dict[str, Any]) -> tuple[Path, Path]:
    cache_root = _canonical_existing_path(
        _required_string(document, "cache_root"), "cache_root"
    )
    temporary_root = _canonical_existing_path(
        _required_string(document, "tmp_root"), "tmp_root"
    )
    if not cache_root.is_dir():
        raise ValueError("cache_root must be a directory")
    if not temporary_root.is_dir():
        raise ValueError("tmp_root must be a directory")
    if _inside(cache_root, temporary_root) or _inside(temporary_root, cache_root):
        raise ValueError("cache_root and tmp_root must not overlap")
    if os.environ.get("TRITON_CACHE_DIR") != str(cache_root):
        raise ValueError("TRITON_CACHE_DIR differs from the controlled cache root")
    if os.environ.get("TMPDIR") != str(temporary_root):
        raise ValueError("TMPDIR differs from the invocation-owned tmp root")
    if os.environ.get("TRITON_JIT_BACKEND") != "NPU":
        raise ValueError("TRITON_JIT_BACKEND must be NPU")
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise ValueError("TORCH_DEVICE_BACKEND_AUTOLOAD must be 0")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("PYTHONDONTWRITEBYTECODE must be 1")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("PYTHONNOUSERSITE must be 1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be 0")
    return cache_root, temporary_root


def _load_standalone(document: dict[str, Any]) -> Any:
    path = _canonical_existing_path(
        _required_string(document, "standalone_compile"),
        "standalone_compile",
    )
    if not path.is_file() or _sha256(path) != _required_sha256(
        document, "standalone_compile_sha256"
    ):
        raise ValueError("standalone compiler identity differs")
    specification = importlib.util.spec_from_file_location(
        "_flagdnn_ascend_standalone_compile", path
    )
    if specification is None or specification.loader is None:
        raise ImportError("cannot load standalone_compile.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(specification.name, None)
        raise
    if not callable(getattr(module, "compile_a_kernel", None)):
        raise RuntimeError("standalone compiler has no compile_a_kernel")
    return module


def _target_manifest() -> dict[str, str]:
    import triton  # Imported only after the cache/environment checks above.

    target = triton.runtime.driver.active.get_current_target()
    backend = str(getattr(target, "backend", ""))
    arch = str(getattr(target, "arch", ""))
    if not backend or not arch:
        raise RuntimeError("active Triton target has no backend/arch identity")
    if backend.lower() not in {"npu", "ascend"}:
        raise RuntimeError(f"active Triton target is not Ascend/NPU: {backend}")
    return {"backend": backend, "arch": arch, "repr": repr(target)}


def _normalized_group_json(path: Path, root: Path) -> str | None:
    if not path.name.startswith("__grp__") or path.suffix != ".json":
        return None
    descriptor = _open_readonly_no_follow(path)
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError(f"group cache file is not regular: {path}")
        encoded_bytes = source.read(MAX_TREE_BYTES + 1)
    if len(encoded_bytes) > MAX_TREE_BYTES:
        raise ValueError("group cache file exceeds attestation byte budget")
    value = _parse_json(encoded_bytes)
    if not isinstance(value, dict):
        raise ValueError(f"group cache file is not an object: {path}")
    child_paths = value.get("child_paths")
    if not isinstance(child_paths, list) or any(
        not isinstance(item, str) for item in child_paths
    ):
        raise ValueError(f"group cache child_paths is invalid: {path}")
    normalized: list[str] = []
    seen: set[str] = set()
    for encoded in child_paths:
        child = Path(encoded)
        if not child.is_absolute():
            raise ValueError("group cache child path must be absolute")
        resolved = child.resolve(strict=True)
        if not _inside(root, resolved):
            raise ValueError("group cache child path escapes cache root")
        relative = resolved.relative_to(root).as_posix()
        if relative in seen:
            raise ValueError("group cache contains a duplicate child path")
        seen.add(relative)
        normalized.append(relative)
    value["child_paths"] = normalized
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    root_text = str(root)
    for key, item in value.items():
        if key != "child_paths" and root_text in json.dumps(item):
            raise ValueError(f"unknown absolute cache-root field in group JSON: {key}")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tree_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        descriptor = path.stat(follow_symlinks=False)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(descriptor.st_mode):
            continue
        if not stat.S_ISREG(descriptor.st_mode):
            raise ValueError(f"cache contains a non-regular entry: {relative}")
        total_bytes += descriptor.st_size
        if len(files) >= MAX_TREE_FILES or total_bytes > MAX_TREE_BYTES:
            raise ValueError("cache tree exceeds attestation budget")
        entry: dict[str, Any] = {
            "path": relative,
            "size": descriptor.st_size,
            "sha256": _sha256(path),
            "mode": stat.S_IMODE(descriptor.st_mode),
        }
        neutral = _normalized_group_json(path, root)
        if neutral is not None:
            entry["root_neutral_codegen_sha256"] = neutral
        files.append(entry)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regular_file_count": len(files),
        "regular_file_bytes": total_bytes,
        "files": files,
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _metadata_workspace(cache_dir: Path, entry_point: str) -> dict[str, Any]:
    metadata_path = cache_dir / f"{entry_point}.json"
    descriptor = metadata_path.stat(follow_symlinks=False)
    if not stat.S_ISREG(descriptor.st_mode):
        raise ValueError("kernel metadata is not a regular file")
    if descriptor.st_size > MAX_TREE_BYTES:
        raise ValueError("kernel metadata exceeds attestation byte budget")
    metadata_descriptor = _open_readonly_no_follow(metadata_path)
    with os.fdopen(metadata_descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != descriptor.st_dev
            or opened.st_ino != descriptor.st_ino
        ):
            raise ValueError("kernel metadata identity changed while opening")
        encoded = source.read(MAX_TREE_BYTES + 1)
    if len(encoded) > MAX_TREE_BYTES:
        raise ValueError("kernel metadata exceeds attestation byte budget")
    metadata = _parse_json(encoded)
    if not isinstance(metadata, dict):
        raise ValueError("kernel metadata must be an object")
    workspace = metadata.get("workspace_size", 0)
    if isinstance(workspace, bool) or not isinstance(workspace, int) or workspace < 0:
        raise ValueError("kernel workspace_size metadata is invalid")
    if workspace != 0:
        raise ValueError("kernel metadata workspace_size must be zero")
    arg_layout = metadata.get("arg_layout")
    if not isinstance(arg_layout, list):
        raise ValueError("kernel metadata is missing arg_layout")
    if len(arg_layout) > MAX_RUNTIME_ARGUMENTS:
        raise ValueError("kernel metadata arg_layout exceeds the runtime ABI limit")
    normalized_layout: list[dict[str, Any]] = []
    for entry in arg_layout:
        if not isinstance(entry, dict):
            raise ValueError("arg_layout entry must be an object")
        kind = entry.get("type")
        mapping = {
            "ptr": "pointer",
            "i32": "i32",
            "i64": "i64",
            "fp32": "f32",
            "fp64": "f64",
        }
        normalized = mapping.get(kind)
        if normalized not in ALLOWED_RUNTIME_TYPES:
            raise ValueError(f"unsupported metadata argument type: {kind}")
        normalized_layout.append({"type": normalized})
    return {
        "relative_path": metadata_path.relative_to(cache_dir).as_posix(),
        "sha256": _sha256(metadata_path),
        "workspace_size": workspace,
        "argument_abi": normalized_layout,
    }


def _compile_candidate(
    document: dict[str, Any], module: Any, cache_root: Path
) -> dict[str, Any]:
    source = _canonical_existing_path(
        _required_string(document, "source_path"), "source_path"
    )
    if not source.is_file() or _sha256(source) != _required_sha256(
        document, "source_sha256"
    ):
        raise ValueError("candidate source identity differs")
    entry_point = _required_string(document, "entry_point")
    if entry_point in {".", ".."} or "/" in entry_point or "\\" in entry_point:
        raise ValueError("entry_point must be a leaf name")
    signature = _required_string(document, "full_signature")
    num_warps = document.get("num_warps")
    num_stages = document.get("num_stages")
    device_ordinal = document.get("device_ordinal")
    for value, name in (
        (num_warps, "num_warps"),
        (num_stages, "num_stages"),
        (device_ordinal, "device_ordinal"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if num_warps <= 0 or num_stages <= 0 or device_ordinal < 0:
        raise ValueError("candidate compile options/device are invalid")
    returned_cache = module.compile_a_kernel(
        str(source),
        entry_point,
        signature,
        num_warps,
        num_stages,
        device_ordinal,
        {},
    )
    if not isinstance(returned_cache, (str, os.PathLike)):
        raise ValueError("standalone compiler cache directory must be path-like")
    cache_dir = _canonical_existing_path(
        os.fspath(returned_cache), "standalone compiler cache directory"
    )
    if not cache_dir.is_dir() or not _inside(cache_root, cache_dir):
        raise ValueError("standalone compiler returned a cache directory outside root")
    return {
        "cache_directory": str(cache_dir),
        "cache_directory_relative": cache_dir.relative_to(cache_root).as_posix(),
        "metadata": _metadata_workspace(cache_dir, entry_point),
    }


def _atomic_output(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    created = False
    published = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        published = True
    except Exception:
        if created and not published:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("driver-bootstrap", "compile-candidate"), required=True
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    output_path = _canonical_output_path(arguments.output)
    request_path = _canonical_existing_path(
        str(arguments.request), "attestation request"
    )
    if output_path == request_path:
        raise ValueError("attestation output must differ from the request")
    request = _read_request(request_path, arguments.mode)
    cache_root, _ = _validate_environment(request)
    if _inside(cache_root, output_path):
        raise ValueError("attestation output must be outside cache_root")
    module = _load_standalone(request)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": arguments.mode,
        "codegen_environment_digest": _required_sha256(
            request, "codegen_environment_digest"
        ),
        "cache_root": str(cache_root),
        "active_target": _target_manifest(),
    }
    if arguments.mode == "compile-candidate":
        result["candidate"] = _compile_candidate(request, module, cache_root)
    else:
        # Accessing the active target imports the pinned driver and builds its
        # npu_utils extension on a fresh cache. The native parent verifies the
        # exact expected extension path/build-id/dependencies separately.
        result["driver_bootstrap"] = {"status": "initialized"}
    result["tree"] = _tree_manifest(cache_root)
    _atomic_output(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
