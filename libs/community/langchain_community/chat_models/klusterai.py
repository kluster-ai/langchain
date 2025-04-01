"""kluster.ai chat models wrapper"""

from __future__ import annotations

import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
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
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
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


logger = logging.getLogger(__name__)



def _convert_message_to_openai(message: BaseMessage):
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            # Handle tool calls according to OpenAI format
            message_dict["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.get("id", f"call_{i}"),
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for i, tc in enumerate(message.tool_calls)
            ]
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


def _parse_tool_calls(raw_tool_calls):
    """Parse tool calls from API response to LangChain format."""
    tool_calls = []
    if not raw_tool_calls:
        return tool_calls

    for tool_call in raw_tool_calls:
        # Check if it's a dict or an object
        if isinstance(tool_call, dict):
            if tool_call.get("type") != "function":
                continue
            function_call = tool_call.get("function", {})
            tool_call_id = tool_call.get("id", "")

            # Get function name and arguments
            if isinstance(function_call, dict):
                function_name = function_call.get("name", "")
                arguments_str = function_call.get("arguments", "{}")
            else:
                function_name = getattr(function_call, "name", "")
                arguments_str = getattr(function_call, "arguments", "{}")
        else:
            # Handle object-like tool_call (e.g., ChatCompletionMessageToolCall)
            if getattr(tool_call, "type", None) != "function":
                continue
            function_call = getattr(tool_call, "function", None)
            tool_call_id = getattr(tool_call, "id", "")

            # Get function name and arguments
            if function_call is None:
                continue
            function_name = getattr(function_call, "name", "")
            arguments_str = getattr(function_call, "arguments", "{}")

        # Parse arguments
        try:
            args = json.loads(arguments_str)
        except json.JSONDecodeError:
            args = arguments_str if isinstance(arguments_str, dict) else {}

        tool_calls.append(
            ToolCall(
                id=tool_call_id,
                name=function_name,
                args=args,
            )
        )
    return tool_calls


class ChatKlusterAI(BaseChatModel):
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
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "Could not import openai python package. "
                    "Please install it with `pip install openai`."
                )
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )

        if self.async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "Could not import openai python package. "
                    "Please install it with `pip install openai`."
                )
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
            additional_kwargs = {}

            # Process function calls if present
            if hasattr(message, "function_call") and message.function_call:
                additional_kwargs["function_call"] = message.function_call

            # Process tool calls if present
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = _parse_tool_calls(message.tool_calls)
                additional_kwargs["tool_calls"] = message.tool_calls

            # Convert OpenAI message back to LangChain message
            if message.role == "assistant":
                lc_message = AIMessage(
                    content=message_content,
                    additional_kwargs=additional_kwargs,
                    tool_calls=tool_calls,
                )
            elif message.role == "user":
                lc_message = HumanMessage(content=message_content)
            elif message.role == "system":
                lc_message = SystemMessage(content=message_content)
            elif message.role == "function":
                lc_message = FunctionMessage(content=message_content, name=message.name)
            elif message.role == "tool":
                lc_message = ToolMessage(
                    content=message_content,
                    tool_call_id=message.tool_call_id
                )
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

            # Handle content chunks
            if choice.delta.content is not None:
                chunk_message = AIMessageChunk(content=choice.delta.content)
                chunk_gen = ChatGenerationChunk(message=chunk_message)

                if run_manager:
                    run_manager.on_llm_new_token(
                        choice.delta.content, chunk=chunk_gen
                    )
                yield chunk_gen

            # Handle tool call chunks
            if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    function_info = tool_call.get("function", {})
                    tool_call_chunk = {
                        "id": tool_call.get("id", ""),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": function_info.get("name", ""),
                            "arguments": function_info.get("arguments", ""),
                        }
                    }

                    chunk_message = AIMessageChunk(
                        content="",
                        additional_kwargs={"tool_calls": [tool_call_chunk]}
                    )
                    chunk_gen = ChatGenerationChunk(message=chunk_message)

                    if run_manager:
                        run_manager.on_llm_new_token("", chunk=chunk_gen)
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

            # Handle content chunks
            if choice.delta.content is not None:
                chunk_message = AIMessageChunk(content=choice.delta.content)
                chunk_gen = ChatGenerationChunk(message=chunk_message)

                if run_manager:
                    await run_manager.on_llm_new_token(
                        choice.delta.content, chunk=chunk_gen
                    )
                yield chunk_gen

            # Handle tool call chunks
            if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    function_info = tool_call.get("function", {})
                    tool_call_chunk = {
                        "id": tool_call.get("id", ""),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": function_info.get("name", ""),
                            "arguments": function_info.get("arguments", ""),
                        }
                    }

                    chunk_message = AIMessageChunk(
                        content="",
                        additional_kwargs={"tool_calls": [tool_call_chunk]}
                    )
                    chunk_gen = ChatGenerationChunk(message=chunk_message)

                    if run_manager:
                        await run_manager.on_llm_new_token("", chunk=chunk_gen)
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to this chat model.

        Args:
            tools: A sequence of tools to bind to this chat model.
            tool_choice: Specifies which tool the model should use. Can be:
                - "auto": model decides whether to call a tool
                - "none": model doesn't call any tools
                - "required": model must call a tool
                - dict: specific tool to use in format {"type": "function", "function": {"name": "tool_name"}}
            **kwargs: Additional parameters to pass to the model.

        Returns:
            A runnable that uses the provided tools.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        tool_names = [ft["function"]["name"] for ft in formatted_tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                if not any(
                    tool_choice["function"]["name"] == name for name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, str):
                chosen = [
                    f for f in formatted_tools if f["function"]["name"] == tool_choice
                ]
                if not chosen:
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, bool):
                if len(formatted_tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be specified when a single tool is "
                        f"passed in. Received {len(tools)} tools."
                    )
                tool_choice = formatted_tools[0]
            else:
                raise ValueError(
                    """Unrecognized tool_choice type. Expected dict having format like
                    this {"type": "function", "function": {"name": <<tool_name>>}}"""
                    f"Received: {tool_choice}"
                )

        kwargs["tool_choice"] = tool_choice
        return super().bind(
            tools=formatted_tools,
            **kwargs
        )
