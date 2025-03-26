"""kluster.ai chat models wrapper"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam
except ImportError:
    raise ImportError(
        "Could not import openai python package. "
        "Please install it with `pip install openai`."
    )

logger = logging.getLogger(__name__)


def _convert_message_to_openai(message: BaseMessage) -> ChatCompletionMessageParam:
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            # Handle tool calls according to OpenAI format
            message_dict["tool_calls"] = message.tool_calls
        return message_dict
    elif isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        return {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise ValueError(f"Got unknown message type: {message}")


class ChatKlusterAi(BaseChatModel):
    """A chat model that uses the kluster.ai API via OpenAI client."""

    model_name: str = Field(default="klusterai/Meta-Llama-3.1-8B-Instruct-Turbo", alias="model")
    """Model name to use."""

    base_url: str = Field(default="https://api.kluster.ai/v1", alias="klusterai_api_base")
    """Base URL for kluster.ai API."""

    api_key: Optional[str] = Field(default=None, alias="klusterai_api_key")
    """API key for kluster.ai."""

    timeout: Optional[Union[float, Tuple[float, float]]] = Field(default=None, alias="request_timeout")
    """Timeout for API requests in seconds."""

    temperature: float = 1.0
    """Sampling temperature between 0 and 2. Higher values like 0.8 will make the output
    more random, while lower values like 0.2 will make it more focused and deterministic."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Additional parameters to pass to the model."""

    top_p: Optional[float] = None
    """Nucleus sampling parameter between 0 and 1."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    frequency_penalty: Optional[float] = None
    """Frequency penalty parameter."""

    presence_penalty: Optional[float] = None
    """Presence penalty parameter."""

    streaming: bool = False
    """Whether to stream responses."""

    max_retries: int = 2
    """Maximum number of retries."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key exists and initialize clients."""
        # Get API key
        self.api_key = get_from_dict_or_env(
            {"api_key": self.api_key},
            "api_key",
            "KLUSTERAI_API_KEY",
            default=None,
        )

        # Also try OPENAI_API_KEY for compatibility
        if self.api_key is None:
            self.api_key = get_from_dict_or_env(
                {},
                "api_key",
                "OPENAI_API_KEY",
                default=None,
            )

        if self.api_key is None:
            raise ValueError(
                "kluster.ai API key not found. Please provide it as klusterai_api_key "
                "or set the KLUSTERAI_API_KEY environment variable."
            )

        # Create clients if not already initialized
        if self.client is None:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

        if self.async_client is None:
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling kluster.ai API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

        if self.max_tokens is not None:
            params["max_completion_tokens"] = self.max_tokens

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts = [_convert_message_to_openai(m) for m in messages]
        params = {
            **self._default_params,
            **kwargs,
        }
        if stop:
            params["stop"] = stop

        response = self.client.chat.completions.create(
            messages=message_dicts,
            **params,
        )
        return self._create_chat_result(response)

    def _create_chat_result(self, response: Any) -> ChatResult:
        generations = []
        for choice in response.choices:
            message = choice.message
            message_content = message.content or ""

            # Convert OpenAI message back to LangChain message
            if message.role == "assistant":
                lc_message = AIMessage(content=message_content)
            elif message.role == "user":
                lc_message = HumanMessage(content=message_content)
            elif message.role == "system":
                lc_message = SystemMessage(content=message_content)
            elif message.role == "function":
                lc_message = FunctionMessage(content=message_content, name=message.name)
            else:
                lc_message = ChatMessage(content=message_content, role=message.role)

            gen = ChatGeneration(
                message=lc_message,
                generation_info=dict(finish_reason=choice.finish_reason),
            )
            generations.append(gen)

        # Extract token usage information
        token_usage = {}
        if hasattr(response, "usage"):
            token_usage = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = [_convert_message_to_openai(m) for m in messages]
        params = {
            **self._default_params,
            "stream": True,
            **kwargs,
        }
        if stop:
            params["stop"] = stop

        for chunk in self.client.chat.completions.create(
            messages=message_dicts,
            **params,
        ):
            if len(chunk.choices) == 0:
                continue

            choice = chunk.choices[0]
            if choice.delta.content is not None:
                chunk_message = AIMessageChunk(content=choice.delta.content)
                chunk_gen = ChatGenerationChunk(message=chunk_message)

                if run_manager:
                    run_manager.on_llm_new_token(
                        choice.delta.content, chunk=chunk_gen
                    )
                yield chunk_gen

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts = [_convert_message_to_openai(m) for m in messages]
        params = {
            **self._default_params,
            **kwargs,
        }
        if stop:
            params["stop"] = stop

        response = await self.async_client.chat.completions.create(
            messages=message_dicts,
            **params,
        )
        return self._create_chat_result(response)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts = [_convert_message_to_openai(m) for m in messages]
        params = {
            **self._default_params,
            "stream": True,
            **kwargs,
        }
        if stop:
            params["stop"] = stop

        async for chunk in await self.async_client.chat.completions.create(
            messages=message_dicts,
            **params,
        ):
            if len(chunk.choices) == 0:
                continue

            choice = chunk.choices[0]
            if choice.delta.content is not None:
                chunk_message = AIMessageChunk(content=choice.delta.content)
                chunk_gen = ChatGenerationChunk(message=chunk_message)

                if run_manager:
                    await run_manager.on_llm_new_token(
                        choice.delta.content, chunk=chunk_gen
                    )
                yield chunk_gen

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "klusterai-chat"

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Any, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a dictionary, pydantic model, callable, or BaseTool.
            **kwargs: Additional parameters to pass to the
                Runnable constructor.
        """
        # OpenAI client automatically handles the tool formatting
        return super().bind(tools=tools, **kwargs)
