"""Exercise run.sh device selection without CUDA or a physical GPU.

Run on Linux: python3 tests/test_run_device_compatibility.py
Only nvidia-smi, the benchmark and optional environment tools are mocked.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUN_SCRIPT = Path(__file__).resolve().parents[1] / "run.sh"


class DeviceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hopper compatibility ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "tools"
        self.build = self.root / "build"
        self.tools.mkdir()
        self.build.mkdir()
        self.output = self.root / "results"
        self.query_log = self.root / "queries.txt"
        self.write_executable(self.tools / "nvidia-smi", r"""#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$MOCK_QUERY_LOG"
for arg in "$@"; do
  case "$arg" in
    --query-gpu=uuid) printf 'GPU-compatibility-test\n'; exit 0 ;;
    --query-gpu=name) printf '%s\n' "$MOCK_GPU_NAME"; exit 0 ;;
    --query-gpu=compute_cap)
      [[ "${MOCK_CAP_FAILURE:-0}" == 0 ]] || exit 1
      printf '%s\n' "$MOCK_COMPUTE_CAP"; exit 0 ;;
  esac
done
printf 'Mock environment query\n'
"""
        )
        for tool in ("lscpu", "ncu", "cuobjdump"):
            self.write_executable(self.tools / tool, "#!/bin/sh\nexit 0\n")
        binary = self.build / "v1"
        self.write_executable(binary, "#!/bin/sh\nexit 0\n")
        self.write_executable(Path(str(binary) + ".validate"), r"""#!/bin/sh
set -eu
[ "$CUDA_VISIBLE_DEVICES" = GPU-compatibility-test ]
printf 'VALIDATOR_EXECUTED\n'
"""
        )
        (self.build / "manifest.tsv").write_text(
            "id\tversion\tcta_m\tcta_n\tcta_k\tpipe\tgmma_n\tcompute_threads\t"
            "producer_threads\tinflight\tepilogue\tbinary\n"
            f"v1\tv1\t128\t128\t64\t3\t128\t128\t0\t1\t0\t{binary}\n"
        )
        (self.build / "build_metadata.txt").write_text("with_validation=1\n")

    @staticmethod
    def write_executable(path, source):
        path.write_text(source)
        path.chmod(0o755)

    def run_script(self, name, capability, mode="validate", dry_run=False, failure=False):
        env = dict(os.environ)
        env.update(
            PATH=str(self.tools) + os.pathsep + env["PATH"],
            MOCK_GPU_NAME=name,
            MOCK_COMPUTE_CAP=capability,
            MOCK_CAP_FAILURE=str(int(failure)),
            MOCK_QUERY_LOG=str(self.query_log),
        )
        args = ["bash", str(RUN_SCRIPT), "--mode", mode, "--build-dir", str(self.build),
                "--output-dir", str(self.output), "--timeout", "0"]
        if dry_run:
            args.append("--dry-run")
        return subprocess.run(args, env=env, text=True, capture_output=True, timeout=30)

    def assert_validation_passes(self, name, capability="9.0"):
        result = self.run_script(name, capability)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("VALIDATOR_EXECUTED", (self.output / "validate_v1.log").read_text())
        metadata = (self.output / "run_metadata.txt").read_text()
        self.assertIn(f"gpu={name}\n", metadata)
        self.assertIn("compute_cap=9.0\n", metadata)

    def test_h800_pcie(self):
        self.assert_validation_passes("NVIDIA H800 PCIe")

    def test_h100(self):
        self.assert_validation_passes("NVIDIA H100 PCIe")

    def test_h200(self):
        self.assert_validation_passes("NVIDIA H200")

    def test_gh200(self):
        self.assert_validation_passes("NVIDIA GH200 480GB")

    def test_model_name_does_not_control_admission(self):
        self.assert_validation_passes("Unlisted Hopper device", " 9.0 ")

    def test_other_architectures_are_rejected(self):
        for index, (name, capability) in enumerate(
                (("NVIDIA A100", "8.0"), ("NVIDIA RTX 3050", "8.6"),
                 ("NVIDIA B200", "10.0"), ("NVIDIA H100 PCIe", "8.0"),
                 ("Unknown device", "9.1"))):
            with self.subTest(name=name, capability=capability):
                output = self.root / f"rejected-{index}"
                self.output = output
                result = self.run_script(name, capability)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("compute capability 9.0", result.stderr)
                self.assertFalse((output / "validate_v1.log").exists())

    def test_unknown_capability_is_rejected(self):
        for index, capability in enumerate(("", "N/A", "9.0\n9.0")):
            with self.subTest(capability=capability):
                self.output = self.root / f"unknown-{index}"
                result = self.run_script("NVIDIA H800 PCIe", capability)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.output / "validate_v1.log").exists())

    def test_capability_query_failure_is_reported(self):
        result = self.run_script("NVIDIA H800 PCIe", "9.0", failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU 计算能力查询失败", result.stderr)
        self.assertFalse((self.output / "validate_v1.log").exists())

    def test_env_accepts_other_architectures_without_capability_query(self):
        result = self.run_script("NVIDIA A100", "8.0", mode="env", failure=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("--query-gpu=compute_cap", self.query_log.read_text())

    def test_disasm_does_not_query_gpu(self):
        result = self.run_script("NVIDIA A100", "8.0", mode="disasm", failure=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.query_log.exists())

    def test_dry_run_does_not_query_gpu_or_create_reports(self):
        result = self.run_script("NVIDIA A100", "8.0", dry_run=True, failure=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.query_log.exists())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
