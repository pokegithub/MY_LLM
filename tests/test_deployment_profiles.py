import subprocess
import sys
import unittest
from pathlib import Path

from config import (
    DEPLOY_16GB_WORKSTATION,
    DEPLOY_2GB_EDGE,
    DEPLOY_8GB_LAPTOP,
    EDGE_0_5B_GQA,
    LOCAL_1_5B_GQA,
    available_deployment_tiers,
    build_deployment_report,
    build_model_config,
    estimate_dense_parameter_count_lower_bound,
    estimate_kv_cache_lower_bound_gb,
    estimate_weight_memory_lower_bound_gb,
    get_deployment_tier,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class DeploymentProfileTests(unittest.TestCase):
    def test_required_deployment_tiers_are_registered(self):
        self.assertEqual(
            set(available_deployment_tiers()),
            {
                DEPLOY_2GB_EDGE,
                DEPLOY_8GB_LAPTOP,
                DEPLOY_16GB_WORKSTATION,
            },
        )

    def test_2gb_tier_is_explicitly_edge_offload_not_3b_claim(self):
        tier = get_deployment_tier(DEPLOY_2GB_EDGE)

        self.assertEqual(tier["target_vram_gb"], 2)
        self.assertEqual(tier["model_profile"], EDGE_0_5B_GQA)
        joined = " ".join(
            list(tier["caveats"]) + [tier["quantization_expectation"]]
        )
        self.assertIn("2GB VRAM does not make a 3B/5B model comfortable", joined)
        self.assertIn("external", joined)
        self.assertIn("unverified", joined)

    def test_deployment_report_does_not_claim_external_runtime_support(self):
        report = build_deployment_report(DEPLOY_16GB_WORKSTATION)

        self.assertEqual(report["model_profile"], LOCAL_1_5B_GQA)
        self.assertFalse(report["profile_is_validated_runtime_support"])
        self.assertEqual(report["repo_verified_external_runtimes"], [])
        self.assertIn("GGUF/llama.cpp", report["external_unverified_targets"])
        self.assertEqual(report["estimates"]["estimate_scope"], "lower_bound_only")
        self.assertGreater(report["estimates"]["weight_memory_lower_bound_gb"], 0)
        self.assertGreater(report["estimates"]["kv_cache_lower_bound_gb"], 0)

    def test_kv_cache_estimate_grows_with_active_context(self):
        cfg = build_model_config(EDGE_0_5B_GQA)

        small = estimate_kv_cache_lower_bound_gb(
            cfg,
            active_context_tokens=512,
        )
        large = estimate_kv_cache_lower_bound_gb(
            cfg,
            active_context_tokens=2048,
        )

        self.assertGreater(large, small)

    def test_weight_estimate_is_positive_lower_bound_for_dense_profile(self):
        profile = EDGE_0_5B_GQA
        cfg = build_model_config(profile)
        params = estimate_dense_parameter_count_lower_bound(cfg)
        weight_gb = estimate_weight_memory_lower_bound_gb(cfg)

        self.assertEqual(cfg.profile_name, profile)
        self.assertIsInstance(params, int)
        self.assertGreater(params, 0)
        self.assertGreater(weight_gb, 0)


class DeploymentInfoCLITests(unittest.TestCase):
    def test_deployment_info_command_is_truthful_about_unverified_runtimes(self):
        result = subprocess.run(
            [sys.executable, "run.py", "deployment-info"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("2gb_edge", result.stdout)
        self.assertIn("8gb_laptop", result.stdout)
        self.assertIn("16gb_workstation", result.stdout)
        self.assertIn("external runtimes   : unverified", result.stdout)
        self.assertIn("lower bounds only", result.stdout)
        self.assertIn("GGUF/GPTQ/AWQ/vLLM/llama.cpp are not implemented", result.stdout)


if __name__ == "__main__":
    unittest.main()
