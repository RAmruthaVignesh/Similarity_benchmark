"""Offline checks for the shared discovery model backends."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from src.llm_backends import create_oci_wayflow_model, create_vllm, validate_reasoning_effort


class LLMBackendTest(unittest.TestCase):
    def test_mistral_backend_preserves_model_loading_options(self):
        vllm = ModuleType("vllm")
        vllm.LLM = Mock(return_value="engine")
        with patch.dict(sys.modules, {"vllm": vllm}):
            result = create_vllm("mistral", "example/model", "revision", max_model_len=2048)
        self.assertEqual(result, "engine")
        vllm.LLM.assert_called_once_with(
            model="example/model", revision="revision", max_model_len=2048,
            tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
        )

    def test_oci_backend_preserves_endpoint_and_reasoning_settings(self):
        wayflow = ModuleType("wayflowcore")
        models = ModuleType("wayflowcore.models")
        client = ModuleType("wayflowcore.models.ociclientconfig")
        models.LlmGenerationConfig = Mock(return_value="generation-config")
        models.OCIGenAIModel = Mock(return_value="oci-model")
        client.OCIClientConfigWithApiKey = Mock(return_value="client-config")
        modules = {"wayflowcore": wayflow, "wayflowcore.models": models,
                   "wayflowcore.models.ociclientconfig": client}
        with patch.dict(sys.modules, modules):
            result = create_oci_wayflow_model(
                model_name="example.model", compartment_id="compartment", config_path="config",
                profile="DEFAULT", region="eu-frankfurt-1", max_tokens=512, seed=7,
                reasoning_effort="high",
            )
        self.assertEqual(result, "oci-model")
        client.OCIClientConfigWithApiKey.assert_called_once_with(
            service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
            auth_profile="DEFAULT", _auth_file_location="config",
        )
        models.LlmGenerationConfig.assert_called_once_with(
            max_tokens=512, extra_args={"seed": 7, "reasoning_effort": "HIGH"},
        )
        models.OCIGenAIModel.assert_called_once_with(
            model_id="example.model", compartment_id="compartment", client_config="client-config",
            generation_config="generation-config",
        )
        validate_reasoning_effort("gpt-6-astra", "high")
