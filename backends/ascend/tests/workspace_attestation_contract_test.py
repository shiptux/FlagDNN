#!/usr/bin/env python3
"""Host-only contract tests for the Ascend workspace attestation helper."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any


DRIVER_KEYS = {
    "schema_version",
    "mode",
    "cache_root",
    "tmp_root",
    "standalone_compile",
    "standalone_compile_sha256",
    "codegen_environment_digest",
}
COMPILE_KEYS = DRIVER_KEYS | {
    "source_path",
    "source_sha256",
    "entry_point",
    "full_signature",
    "num_warps",
    "num_stages",
    "device_ordinal",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class AttestationFixture:
    def __init__(self, script: Path) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix="flagdnn-ascend-attestation-contract-"
        )
        self.root = Path(self._temporary.name).resolve(strict=True)
        self.script = script
        self.cache = self.root / "cache"
        self.tmp = self.root / "tmp"
        self.modules = self.root / "modules"
        self.compile_cache = self.cache / "compiled-kernel"
        self.cache.mkdir()
        self.tmp.mkdir()
        self.modules.mkdir()
        self.compile_cache.mkdir()

        self.source = self.root / "kernel.py"
        self.source.write_text("def add_kernel():\n    pass\n", encoding="utf-8")
        self.standalone = self.root / "standalone_compile.py"
        self.import_marker = self.root / "standalone-imported"
        self.compile_marker = self.root / "compile-called"
        self.request_path = self.root / "request.json"
        self.output_path = self.root / "result.json"
        self.victim_path = self.root / "must-not-change"
        self.entry_point = "add_kernel"
        self.signature = "*fp32:16,*fp32:16,*fp32:16,i32"
        self.digest = "a" * 64

        self._write_fake_triton()
        self._write_fake_standalone()
        (self.cache / "driver.bin").write_bytes(b"fake-npu-utils")
        self.write_metadata()

        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PYTHONPATH": str(self.modules),
                "TRITON_CACHE_DIR": str(self.cache),
                "TMPDIR": str(self.tmp),
                "TRITON_JIT_BACKEND": "NPU",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
                "FLAGDNN_FAKE_CACHE_DIRECTORY": str(self.compile_cache),
                "FLAGDNN_FAKE_SOURCE": str(self.source),
                "FLAGDNN_FAKE_ENTRY_POINT": self.entry_point,
                "FLAGDNN_FAKE_SIGNATURE": self.signature,
                "FLAGDNN_FAKE_IMPORT_MARKER": str(self.import_marker),
                "FLAGDNN_FAKE_COMPILE_MARKER": str(self.compile_marker),
            }
        )

    def close(self) -> None:
        self._temporary.cleanup()

    def _write_fake_triton(self) -> None:
        package = self.modules / "triton"
        package.mkdir()
        (package / "__init__.py").write_text(
            """
class _Target:
    backend = "NPU"
    arch = "Ascend910B1"

    def __repr__(self):
        return "FakeAscendTarget(arch=Ascend910B1)"


class _Active:
    @staticmethod
    def get_current_target():
        return _Target()


class _Driver:
    active = _Active()


class _Runtime:
    driver = _Driver()


runtime = _Runtime()
""".lstrip(),
            encoding="utf-8",
        )

    def _write_fake_standalone(self) -> None:
        self.standalone.write_text(
            """
import os
from pathlib import Path


marker = os.environ.get("FLAGDNN_FAKE_IMPORT_MARKER")
if marker:
    Path(marker).write_text("imported", encoding="utf-8")


def compile_a_kernel(source, entry_point, signature, num_warps,
                     num_stages, device_ordinal, extra):
    expected = (
        os.environ["FLAGDNN_FAKE_SOURCE"],
        os.environ["FLAGDNN_FAKE_ENTRY_POINT"],
        os.environ["FLAGDNN_FAKE_SIGNATURE"],
        4,
        2,
        0,
        {},
    )
    actual = (
        source,
        entry_point,
        signature,
        num_warps,
        num_stages,
        device_ordinal,
        extra,
    )
    if actual != expected:
        raise RuntimeError(f"unexpected fake compile request: {actual!r}")
    Path(os.environ["FLAGDNN_FAKE_COMPILE_MARKER"]).write_text(
        "called", encoding="utf-8"
    )
    return os.environ["FLAGDNN_FAKE_CACHE_DIRECTORY"]
""".lstrip(),
            encoding="utf-8",
        )

    def write_metadata(
        self,
        *,
        workspace: object = 0,
        include_workspace: bool = True,
        arg_layout: object | None = None,
        entry_point: str | None = None,
    ) -> Path:
        metadata: dict[str, object] = {
            "arg_layout": (
                [
                    {"type": "ptr", "dtype": "fp32"},
                    {"type": "i32"},
                    {"type": "i64"},
                    {"type": "fp32"},
                    {"type": "fp64"},
                ]
                if arg_layout is None
                else arg_layout
            )
        }
        if include_workspace:
            metadata["workspace_size"] = workspace
        path = self.compile_cache / f"{entry_point or self.entry_point}.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def request(self, mode: str) -> dict[str, object]:
        request: dict[str, object] = {
            "schema_version": 1,
            "mode": mode,
            "cache_root": str(self.cache),
            "tmp_root": str(self.tmp),
            "standalone_compile": str(self.standalone),
            "standalone_compile_sha256": _sha256(self.standalone),
            "codegen_environment_digest": self.digest,
        }
        if mode == "compile-candidate":
            request.update(
                {
                    "source_path": str(self.source),
                    "source_sha256": _sha256(self.source),
                    "entry_point": self.entry_point,
                    "full_signature": self.signature,
                    "num_warps": 4,
                    "num_stages": 2,
                    "device_ordinal": 0,
                }
            )
        return request

    def write_request(
        self, document: dict[str, object], path: Path | None = None
    ) -> Path:
        destination = path or self.request_path
        destination.write_bytes(_canonical_json(document))
        return destination

    def invoke(
        self,
        mode: str,
        *,
        document: dict[str, object] | None = None,
        request_path: Path | str | None = None,
        output_path: Path | str | None = None,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if document is not None:
            self.write_request(document)
        elif request_path is None and not self.request_path.exists():
            self.write_request(self.request(mode))
        command = [
            sys.executable,
            str(self.script),
            "--mode",
            mode,
            "--request",
            str(request_path or self.request_path),
            "--output",
            str(output_path or self.output_path),
        ]
        return subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment or self.environment,
            cwd=cwd or self.root,
            timeout=20,
        )


class WorkspaceAttestationContractTest(unittest.TestCase):
    script: Path

    @classmethod
    def setUpClass(cls) -> None:
        if len(sys.argv) != 2:
            raise RuntimeError(
                "usage: workspace_attestation_contract_test.py ATTESTATION_SCRIPT"
            )
        cls.script = Path(sys.argv[1]).resolve(strict=True)

    def setUp(self) -> None:
        self.fixture = AttestationFixture(self.script)

    def tearDown(self) -> None:
        self.fixture.close()

    def assert_success(
        self, completed: subprocess.CompletedProcess[str]
    ) -> dict[str, Any]:
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        result = json.loads(self.fixture.output_path.read_bytes())
        self.assertTrue(self.fixture.output_path.read_bytes().endswith(b"\n"))
        temporary_pattern = f".{self.fixture.output_path.name}.tmp-*"
        self.assertEqual(list(self.fixture.output_path.parent.glob(temporary_pattern)), [])
        return result

    def assert_failure(
        self,
        completed: subprocess.CompletedProcess[str],
        message: str,
        *,
        output_path: Path | None = None,
        previous: bytes | None = None,
    ) -> None:
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(message, completed.stderr)
        destination = output_path or self.fixture.output_path
        if previous is None:
            self.assertFalse(destination.exists())
        else:
            self.assertEqual(destination.read_bytes(), previous)

    def assert_tree_integrity(self, tree: dict[str, Any]) -> None:
        unsigned = copy.deepcopy(tree)
        manifest_hash = unsigned.pop("manifest_sha256")
        self.assertEqual(manifest_hash, hashlib.sha256(_canonical_json(unsigned)).hexdigest())
        self.assertEqual(unsigned["regular_file_count"], len(unsigned["files"]))
        self.assertEqual(
            unsigned["regular_file_bytes"],
            sum(entry["size"] for entry in unsigned["files"]),
        )
        for entry in unsigned["files"]:
            path = self.fixture.cache / entry["path"]
            self.assertEqual(entry["sha256"], _sha256(path))
            self.assertFalse(Path(entry["path"]).is_absolute())
            self.assertNotIn("..", Path(entry["path"]).parts)

    def test_driver_bootstrap_schema_and_manifest(self) -> None:
        completed = self.fixture.invoke(
            "driver-bootstrap", document=self.fixture.request("driver-bootstrap")
        )
        result = self.assert_success(completed)
        self.assertEqual(
            set(result),
            {
                "schema_version",
                "mode",
                "codegen_environment_digest",
                "cache_root",
                "active_target",
                "driver_bootstrap",
                "tree",
            },
        )
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["mode"], "driver-bootstrap")
        self.assertEqual(result["codegen_environment_digest"], self.fixture.digest)
        self.assertEqual(result["cache_root"], str(self.fixture.cache))
        self.assertEqual(
            result["active_target"],
            {
                "backend": "NPU",
                "arch": "Ascend910B1",
                "repr": "FakeAscendTarget(arch=Ascend910B1)",
            },
        )
        self.assertEqual(result["driver_bootstrap"], {"status": "initialized"})
        self.assert_tree_integrity(result["tree"])
        self.assertTrue(self.fixture.import_marker.is_file())
        self.assertFalse(self.fixture.compile_marker.exists())

    def test_compile_candidate_schema_metadata_and_fake_call(self) -> None:
        completed = self.fixture.invoke(
            "compile-candidate", document=self.fixture.request("compile-candidate")
        )
        result = self.assert_success(completed)
        self.assertEqual(
            set(result),
            {
                "schema_version",
                "mode",
                "codegen_environment_digest",
                "cache_root",
                "active_target",
                "candidate",
                "tree",
            },
        )
        candidate = result["candidate"]
        self.assertEqual(candidate["cache_directory"], str(self.fixture.compile_cache))
        self.assertEqual(candidate["cache_directory_relative"], "compiled-kernel")
        metadata_path = self.fixture.compile_cache / f"{self.fixture.entry_point}.json"
        self.assertEqual(
            candidate["metadata"],
            {
                "relative_path": f"{self.fixture.entry_point}.json",
                "sha256": _sha256(metadata_path),
                "workspace_size": 0,
                "argument_abi": [
                    {"type": "pointer"},
                    {"type": "i32"},
                    {"type": "i64"},
                    {"type": "f32"},
                    {"type": "f64"},
                ],
            },
        )
        self.assert_tree_integrity(result["tree"])
        self.assertTrue(self.fixture.import_marker.is_file())
        self.assertTrue(self.fixture.compile_marker.is_file())

    def test_both_request_modes_have_exact_required_schema(self) -> None:
        for mode, keys in (
            ("driver-bootstrap", DRIVER_KEYS),
            ("compile-candidate", COMPILE_KEYS),
        ):
            baseline = self.fixture.request(mode)
            self.assertEqual(set(baseline), keys)
            for key in sorted(keys):
                if key in {"schema_version", "mode"}:
                    continue
                with self.subTest(mode=mode, missing=key):
                    request = copy.deepcopy(baseline)
                    del request[key]
                    completed = self.fixture.invoke(mode, document=request)
                    self.assert_failure(completed, "request schema")
            with self.subTest(mode=mode, unknown="unexpected"):
                request = copy.deepcopy(baseline)
                request["unexpected"] = "must-not-be-ignored"
                completed = self.fixture.invoke(mode, document=request)
                self.assert_failure(completed, "request schema")

    def test_schema_version_mode_and_json_are_strict(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")
        cases: list[tuple[str, bytes, str]] = []
        wrong_version = copy.deepcopy(baseline)
        wrong_version["schema_version"] = 2
        cases.append(("version", _canonical_json(wrong_version), "schema is unsupported"))
        boolean_version = copy.deepcopy(baseline)
        boolean_version["schema_version"] = True
        cases.append(("boolean-version", _canonical_json(boolean_version), "schema is unsupported"))
        wrong_mode = copy.deepcopy(baseline)
        wrong_mode["mode"] = "compile-candidate"
        cases.append(("mode", _canonical_json(wrong_mode), "mode differs"))
        duplicate = _canonical_json(baseline)[:-1] + b',"mode":"driver-bootstrap"}'
        cases.append(("duplicate", duplicate, "duplicate JSON key"))
        for name, encoded, message in cases:
            with self.subTest(name=name):
                self.fixture.request_path.write_bytes(encoded)
                completed = self.fixture.invoke("driver-bootstrap")
                self.assert_failure(completed, message)

    def test_compile_scalar_schema_rejects_bool_noninteger_and_range(self) -> None:
        baseline = self.fixture.request("compile-candidate")
        for key in ("num_warps", "num_stages", "device_ordinal"):
            bad_values: tuple[object, ...] = (True, "1", 1.0)
            if key == "device_ordinal":
                bad_values += (-1,)
            else:
                bad_values += (0, -1)
            for value in bad_values:
                with self.subTest(key=key, value=value):
                    request = copy.deepcopy(baseline)
                    request[key] = value
                    completed = self.fixture.invoke("compile-candidate", document=request)
                    expected = (
                        key if isinstance(value, (str, float, bool)) else "invalid"
                    )
                    self.assert_failure(completed, expected)

    def test_environment_is_exact_and_checked_before_import(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")
        wrong_values = {
            "TRITON_CACHE_DIR": str(self.fixture.root),
            "TMPDIR": str(self.fixture.root),
            "TRITON_JIT_BACKEND": "CPU",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "0",
            "PYTHONNOUSERSITE": "0",
            "PYTHONHASHSEED": "1",
        }
        for key, value in wrong_values.items():
            with self.subTest(key=key):
                environment = self.fixture.environment.copy()
                environment[key] = value
                completed = self.fixture.invoke(
                    "driver-bootstrap", document=baseline, environment=environment
                )
                self.assert_failure(completed, key)
                self.assertFalse(self.fixture.import_marker.exists())

    def test_cache_and_tmp_are_distinct_non_overlapping_directories(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")

        not_a_directory = self.fixture.root / "not-a-directory"
        not_a_directory.write_bytes(b"regular-file")
        request = copy.deepcopy(baseline)
        request["tmp_root"] = str(not_a_directory)
        environment = self.fixture.environment.copy()
        environment["TMPDIR"] = str(not_a_directory)
        completed = self.fixture.invoke(
            "driver-bootstrap", document=request, environment=environment
        )
        self.assert_failure(completed, "tmp_root must be a directory")
        self.assertFalse(self.fixture.import_marker.exists())

        request = copy.deepcopy(baseline)
        request["tmp_root"] = str(self.fixture.cache)
        environment = self.fixture.environment.copy()
        environment["TMPDIR"] = str(self.fixture.cache)
        completed = self.fixture.invoke(
            "driver-bootstrap", document=request, environment=environment
        )
        self.assert_failure(completed, "must not overlap")
        self.assertFalse(self.fixture.import_marker.exists())

        nested_tmp = self.fixture.cache / "nested-tmp"
        nested_tmp.mkdir()
        request = copy.deepcopy(baseline)
        request["tmp_root"] = str(nested_tmp)
        environment = self.fixture.environment.copy()
        environment["TMPDIR"] = str(nested_tmp)
        completed = self.fixture.invoke(
            "driver-bootstrap", document=request, environment=environment
        )
        self.assert_failure(completed, "must not overlap")
        self.assertFalse(self.fixture.import_marker.exists())

    def test_request_and_declared_paths_must_be_absolute_canonical_and_no_follow(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")

        real_request = self.fixture.root / "real-request.json"
        self.fixture.write_request(baseline, real_request)
        request_link = self.fixture.root / "request-link.json"
        request_link.symlink_to(real_request)
        completed = self.fixture.invoke(
            "driver-bootstrap", request_path=request_link
        )
        self.assert_failure(completed, "attestation request must already be canonical")

        completed = self.fixture.invoke(
            "driver-bootstrap",
            request_path=real_request.name,
            cwd=self.fixture.root,
        )
        self.assert_failure(completed, "attestation request must be absolute")

        cache_link = self.fixture.root / "cache-link"
        cache_link.symlink_to(self.fixture.cache, target_is_directory=True)
        request = copy.deepcopy(baseline)
        request["cache_root"] = str(cache_link)
        environment = self.fixture.environment.copy()
        environment["TRITON_CACHE_DIR"] = str(cache_link)
        completed = self.fixture.invoke(
            "driver-bootstrap", document=request, environment=environment
        )
        self.assert_failure(completed, "cache_root must already be canonical")

        request = copy.deepcopy(baseline)
        (self.fixture.cache / "child").mkdir()
        request["cache_root"] = str(self.fixture.cache / "child" / "..")
        environment = self.fixture.environment.copy()
        environment["TRITON_CACHE_DIR"] = str(request["cache_root"])
        completed = self.fixture.invoke(
            "driver-bootstrap", document=request, environment=environment
        )
        self.assert_failure(completed, "cache_root must already be canonical")

        standalone_link = self.fixture.root / "standalone-link.py"
        standalone_link.symlink_to(self.fixture.standalone)
        request = copy.deepcopy(baseline)
        request["standalone_compile"] = str(standalone_link)
        completed = self.fixture.invoke("driver-bootstrap", document=request)
        self.assert_failure(completed, "standalone_compile must already be canonical")

    def test_source_and_returned_cache_paths_reject_traversal_and_symlinks(self) -> None:
        baseline = self.fixture.request("compile-candidate")
        source_link = self.fixture.root / "source-link.py"
        source_link.symlink_to(self.fixture.source)
        request = copy.deepcopy(baseline)
        request["source_path"] = str(source_link)
        completed = self.fixture.invoke("compile-candidate", document=request)
        self.assert_failure(completed, "source_path must already be canonical")
        self.assertFalse(self.fixture.compile_marker.exists())

        request = copy.deepcopy(baseline)
        (self.fixture.root / "subdir").mkdir()
        request["source_path"] = str(self.fixture.root / "subdir" / ".." / "kernel.py")
        completed = self.fixture.invoke("compile-candidate", document=request)
        self.assert_failure(completed, "source_path must already be canonical")
        self.assertFalse(self.fixture.compile_marker.exists())

        cache_link = self.fixture.root / "returned-cache-link"
        cache_link.symlink_to(self.fixture.compile_cache, target_is_directory=True)
        environment = self.fixture.environment.copy()
        environment["FLAGDNN_FAKE_CACHE_DIRECTORY"] = str(cache_link)
        completed = self.fixture.invoke(
            "compile-candidate", document=baseline, environment=environment
        )
        self.assert_failure(completed, "cache directory must already be canonical")

        outside = self.fixture.root / "outside-cache"
        outside.mkdir()
        self.fixture.write_metadata(entry_point=self.fixture.entry_point)
        environment = self.fixture.environment.copy()
        environment["FLAGDNN_FAKE_CACHE_DIRECTORY"] = str(outside)
        completed = self.fixture.invoke(
            "compile-candidate", document=baseline, environment=environment
        )
        self.assert_failure(completed, "outside root")

    def test_entry_point_cannot_escape_metadata_directory(self) -> None:
        request = self.fixture.request("compile-candidate")
        request["entry_point"] = "../outside"
        self.fixture.environment["FLAGDNN_FAKE_ENTRY_POINT"] = "../outside"
        outside_metadata = self.fixture.cache / "outside.json"
        outside_metadata.write_text(
            json.dumps({"workspace_size": 0, "arg_layout": [{"type": "ptr"}]}),
            encoding="utf-8",
        )
        completed = self.fixture.invoke("compile-candidate", document=request)
        self.assert_failure(completed, "entry_point must be a leaf name")
        self.assertFalse(self.fixture.compile_marker.exists())

    def test_cache_tree_rejects_symlinks_and_group_path_escape(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")
        outside = self.fixture.root / "outside.bin"
        outside.write_bytes(b"outside")
        cache_link = self.fixture.cache / "linked.bin"
        cache_link.symlink_to(outside)
        completed = self.fixture.invoke("driver-bootstrap", document=baseline)
        self.assert_failure(completed, "non-regular entry")
        cache_link.unlink()

        group = self.fixture.cache / "__grp__escape.json"
        group.write_text(
            json.dumps({"child_paths": [str(outside)]}), encoding="utf-8"
        )
        completed = self.fixture.invoke("driver-bootstrap", document=baseline)
        self.assert_failure(completed, "escapes cache root")
        group.unlink()

        child = self.fixture.cache / "driver.bin"
        group.write_text(
            json.dumps({"child_paths": [str(child), str(child)]}), encoding="utf-8"
        )
        completed = self.fixture.invoke("driver-bootstrap", document=baseline)
        self.assert_failure(completed, "duplicate child path")
        group.unlink()

        group.write_text(
            json.dumps(
                {
                    "child_paths": [str(child)],
                    "unknown_root": str(self.fixture.cache),
                }
            ),
            encoding="utf-8",
        )
        completed = self.fixture.invoke("driver-bootstrap", document=baseline)
        self.assert_failure(completed, "unknown absolute cache-root field")

    def test_group_manifest_normalizes_cache_root(self) -> None:
        neutral_hashes: list[str] = []
        for fixture in (self.fixture, AttestationFixture(self.script)):
            try:
                child = fixture.cache / "driver.bin"
                group = fixture.cache / "__grp__kernel.json"
                group.write_text(
                    json.dumps(
                        {"child_paths": [str(child)], "kind": "fake-group"}
                    ),
                    encoding="utf-8",
                )
                completed = fixture.invoke(
                    "driver-bootstrap", document=fixture.request("driver-bootstrap")
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                output = json.loads(fixture.output_path.read_bytes())
                entry = next(
                    item
                    for item in output["tree"]["files"]
                    if item["path"] == "__grp__kernel.json"
                )
                neutral_hashes.append(entry["root_neutral_codegen_sha256"])
            finally:
                if fixture is not self.fixture:
                    fixture.close()
        self.assertEqual(neutral_hashes[0], neutral_hashes[1])

    def test_request_file_count_and_byte_budgets_fail_before_publication(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")

        oversized_request = self.fixture.root / "oversized-request.json"
        with oversized_request.open("wb") as stream:
            stream.truncate((1 << 20) + 1)
        completed = self.fixture.invoke(
            "driver-bootstrap", request_path=oversized_request
        )
        self.assert_failure(completed, "bounded regular file")

        for index in range(4097):
            (self.fixture.cache / f"entry-{index:04d}").touch()
        completed = self.fixture.invoke("driver-bootstrap", document=baseline)
        self.assert_failure(completed, "tree exceeds attestation budget")
        for path in self.fixture.cache.glob("entry-*"):
            path.unlink()

        oversized_cache_file = self.fixture.cache / "oversized-cache-file"
        with oversized_cache_file.open("wb") as stream:
            stream.truncate((1 << 30) + 1)
        completed = self.fixture.invoke("driver-bootstrap", document=baseline)
        self.assert_failure(completed, "tree exceeds attestation budget")

    def test_metadata_workspace_is_exactly_zero(self) -> None:
        baseline = self.fixture.request("compile-candidate")
        for workspace in (1, 4096):
            with self.subTest(workspace=workspace):
                self.fixture.write_metadata(workspace=workspace)
                completed = self.fixture.invoke("compile-candidate", document=baseline)
                self.assert_failure(completed, "workspace_size must be zero")

        for workspace in (True, -1, 1.5, "0"):
            with self.subTest(invalid=workspace):
                self.fixture.write_metadata(workspace=workspace)
                completed = self.fixture.invoke("compile-candidate", document=baseline)
                self.assert_failure(completed, "workspace_size metadata is invalid")

        self.fixture.write_metadata(include_workspace=False)
        completed = self.fixture.invoke("compile-candidate", document=baseline)
        result = self.assert_success(completed)
        self.assertEqual(result["candidate"]["metadata"]["workspace_size"], 0)

    def test_metadata_argument_layout_is_typed_and_bounded(self) -> None:
        baseline = self.fixture.request("compile-candidate")
        invalid_layouts: tuple[object, ...] = (
            "ptr",
            {},
            ["ptr"],
            [{}],
            [{"type": "u32"}],
        )
        for layout in invalid_layouts:
            with self.subTest(layout=layout):
                self.fixture.write_metadata(arg_layout=layout)
                completed = self.fixture.invoke("compile-candidate", document=baseline)
                if not isinstance(layout, list):
                    expected = "arg_layout"
                elif layout and isinstance(layout[0], dict):
                    expected = "argument type"
                else:
                    expected = "entry must be an object"
                self.assert_failure(completed, expected)

        self.fixture.write_metadata(arg_layout=[{"type": "i32"}] * 4097)
        completed = self.fixture.invoke("compile-candidate", document=baseline)
        self.assert_failure(completed, "runtime ABI limit")

        metadata = self.fixture.compile_cache / f"{self.fixture.entry_point}.json"
        metadata.unlink()
        metadata.symlink_to(self.fixture.cache / "driver.bin")
        completed = self.fixture.invoke("compile-candidate", document=baseline)
        self.assert_failure(completed, "metadata is not a regular file")

    def test_identity_hash_tampering_never_publishes(self) -> None:
        driver = self.fixture.request("driver-bootstrap")
        driver["standalone_compile_sha256"] = "0" * 64
        completed = self.fixture.invoke("driver-bootstrap", document=driver)
        self.assert_failure(completed, "standalone compiler identity differs")
        self.assertFalse(self.fixture.import_marker.exists())

        candidate = self.fixture.request("compile-candidate")
        candidate["source_sha256"] = "0" * 64
        completed = self.fixture.invoke("compile-candidate", document=candidate)
        self.assert_failure(completed, "candidate source identity differs")
        self.assertFalse(self.fixture.compile_marker.exists())

        driver = self.fixture.request("driver-bootstrap")
        driver["codegen_environment_digest"] = "A" * 64
        completed = self.fixture.invoke("driver-bootstrap", document=driver)
        self.assert_failure(completed, "lowercase SHA-256")

    def test_output_path_is_canonical_no_follow_and_failure_preserves_old_result(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")
        self.fixture.victim_path.write_bytes(b"victim-must-survive")
        output_link = self.fixture.root / "output-link.json"
        output_link.symlink_to(self.fixture.victim_path)
        completed = self.fixture.invoke(
            "driver-bootstrap", document=baseline, output_path=output_link
        )
        self.assert_failure(
            completed,
            "attestation output must not be a symlink",
            output_path=self.fixture.victim_path,
            previous=b"victim-must-survive",
        )
        self.assertTrue(output_link.is_symlink())
        self.assertFalse(self.fixture.import_marker.exists())

        completed = self.fixture.invoke(
            "driver-bootstrap",
            document=baseline,
            output_path="relative-result.json",
            cwd=self.fixture.root,
        )
        self.assert_failure(
            completed,
            "attestation output must be absolute",
            output_path=self.fixture.root / "relative-result.json",
        )

        old_result = b"previous-committed-result\n"
        self.fixture.output_path.write_bytes(old_result)
        bad = copy.deepcopy(baseline)
        bad["standalone_compile_sha256"] = "0" * 64
        completed = self.fixture.invoke("driver-bootstrap", document=bad)
        self.assert_failure(completed, "identity differs", previous=old_result)

    def test_output_cannot_overwrite_request_or_invalidate_cache_manifest(self) -> None:
        baseline = self.fixture.request("driver-bootstrap")
        request_bytes = _canonical_json(baseline)
        self.fixture.request_path.write_bytes(request_bytes)
        completed = self.fixture.invoke(
            "driver-bootstrap",
            request_path=self.fixture.request_path,
            output_path=self.fixture.request_path,
        )
        self.assert_failure(
            completed,
            "must differ from the request",
            output_path=self.fixture.request_path,
            previous=request_bytes,
        )
        self.assertFalse(self.fixture.import_marker.exists())

        cache_output = self.fixture.cache / "result.json"
        completed = self.fixture.invoke(
            "driver-bootstrap", document=baseline, output_path=cache_output
        )
        self.assert_failure(
            completed,
            "must be outside cache_root",
            output_path=cache_output,
        )
        self.assertFalse(self.fixture.import_marker.exists())

    def test_atomic_replace_preserves_old_result_on_pre_publish_failure(self) -> None:
        specification = importlib.util.spec_from_file_location(
            "_flagdnn_attestation_atomic_contract", self.script
        )
        self.assertIsNotNone(specification)
        self.assertIsNotNone(specification.loader if specification else None)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)

        output = self.fixture.root / "atomic-result.json"
        old_result = b"old-result\n"
        output.write_bytes(old_result)
        process_id = 424242
        temporary = output.with_name(f".{output.name}.tmp-{process_id}")

        temporary.symlink_to(self.fixture.victim_path)
        self.fixture.victim_path.write_bytes(b"temp-victim")
        with mock.patch.object(module.os, "getpid", return_value=process_id):
            with self.assertRaises(FileExistsError):
                module._atomic_output(output, {"schema_version": 1})
        self.assertEqual(output.read_bytes(), old_result)
        self.assertEqual(self.fixture.victim_path.read_bytes(), b"temp-victim")
        self.assertTrue(temporary.is_symlink())
        temporary.unlink()

        with mock.patch.object(module.os, "getpid", return_value=process_id):
            with mock.patch.object(module.os, "replace", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    module._atomic_output(output, {"schema_version": 1})
        self.assertEqual(output.read_bytes(), old_result)
        self.assertFalse(temporary.exists())


def main() -> int:
    script_argument = sys.argv[1:]
    if len(script_argument) != 1:
        raise RuntimeError(
            "usage: workspace_attestation_contract_test.py ATTESTATION_SCRIPT"
        )
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        WorkspaceAttestationContractTest
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
