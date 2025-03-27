from unittest.mock import patch
import pytest

from langchain_community.chat_models import ChatKlusterAi


@pytest.mark.requires("openai")
def test_klusterai_model_param() -> None:
    test_cases = [
        {"model_name": "foo", "api_key": "test_key"},
        {"model": "foo", "api_key": "test_key"},
        {"model_name": "foo", "klusterai_api_key": "test_key"},
        {"model": "foo", "klusterai_api_key": "test_key"},
    ]

    for case in test_cases:
        with patch.object(ChatKlusterAi, "validate_environment", return_value=None):
            llm = ChatKlusterAi(**case)
            assert llm.model_name == "foo"
            if "api_key" in case:
                assert llm.api_key == "test_key"
            else:
                assert llm.api_key == "test_key"


@pytest.mark.requires("openai")
def test_klusterai_model_default_params() -> None:
    with patch.object(ChatKlusterAi, "validate_environment", return_value=None):
        llm = ChatKlusterAi(api_key="test_key")
        assert llm.model_name == "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo"
        assert llm.base_url == "https://api.kluster.ai/v1"
        assert llm.api_key == "test_key"
        assert llm.temperature == 1.0
