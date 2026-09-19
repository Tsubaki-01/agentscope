# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for OllamaChatModel with mocked API responses.

Tests cover both non-streaming and streaming modes.
Ollama uses ollama.AsyncClient with async iterator streaming.
"""
import importlib.util
import json
from datetime import datetime
from typing import Any
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from utils import AnyString

from agentscope.agent import Agent, InjectionConfig, ReActConfig
from agentscope.message import (
    TextBlock,
    ToolCallBlock,
    ThinkingBlock,
    ToolResultBlock,
    UserMsg,
)
from agentscope.model import OllamaChatModel
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit, ToolChunk, ToolChoice
from agentscope.types import ReplyFinishedReason

A = AnyString()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(stream: bool = False) -> Any:
    return OllamaChatModel(
        model="qwen3:8b",
        stream=stream,
        context_size=40_960,
    )


def _mock_completion(
    content: str = "",
    thinking: str | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Build a mock non-streaming Ollama response."""
    msg = MagicMock()
    msg.content = content
    msg.thinking = thinking

    if tool_calls:
        tc_mocks = []
        for tc in tool_calls:
            m = MagicMock()
            m.function.name = tc["name"]
            m.function.arguments = tc["args"]
            tc_mocks.append(m)
        msg.tool_calls = tc_mocks
    else:
        msg.tool_calls = None

    resp = MagicMock()
    resp.message = msg
    resp.prompt_eval_count = 10
    resp.eval_count = 5
    resp.id = None
    return resp


def _make_stream_chunk(
    content: str = "",
    thinking: str | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Build a single mock Ollama streaming chunk."""
    msg = MagicMock()
    msg.content = content
    msg.thinking = thinking

    if tool_calls:
        tc_mocks = []
        for tc in tool_calls:
            m = MagicMock()
            m.function.name = tc["name"]
            m.function.arguments = tc["args"]
            tc_mocks.append(m)
        msg.tool_calls = tc_mocks
    else:
        msg.tool_calls = None

    chunk = MagicMock()
    chunk.message = msg
    chunk.prompt_eval_count = 10
    chunk.eval_count = 5
    chunk.id = None
    return chunk


class _MockAsyncStream:
    """Mock async iterator for Ollama stream."""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "_MockAsyncStream":
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


class TestOllamaNonStream(IsolatedAsyncioTestCase):
    """Tests for OllamaChatModel in non-streaming mode."""

    def setUp(self) -> None:
        self.model = _make_model(stream=False)
        # Client is built eagerly in __init__; inject a mock onto the
        # instance so chat() hits it instead of the network.
        self.mock_client = MagicMock()
        self.model.client = self.mock_client

    async def test_text_response(self) -> None:
        """Non-stream text response returns a single ChatResponse."""
        self.mock_client.chat = AsyncMock(
            return_value=_mock_completion(content="Hello!"),
        )

        result = await self.model([])

        self.assertEqual(
            (result.is_last, result.content),
            (
                True,
                [TextBlock.model_construct(id=A, created_at=A, text="Hello!")],
            ),
        )

    async def test_tool_call_response(self) -> None:
        """Parsing a tool-call response creates a ToolCallBlock."""
        self.mock_client.chat = AsyncMock(
            return_value=_mock_completion(
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "SH"}},
                ],
            ),
        )

        result = await self.model([])

        self.assertEqual(
            (result.is_last, result.content),
            (
                True,
                [
                    ToolCallBlock.model_construct(
                        id=A,
                        created_at=A,
                        name="get_weather",
                        input=json.dumps({"city": "SH"}),
                    ),
                ],
            ),
        )

    async def test_thinking_response(self) -> None:
        """Non-stream thinking plus text returns ThinkingBlock then
        TextBlock."""
        self.mock_client.chat = AsyncMock(
            return_value=_mock_completion(
                content="Answer",
                thinking="Let me think...",
            ),
        )

        result = await self.model([])

        self.assertEqual(
            (result.is_last, result.content),
            (
                True,
                [
                    ThinkingBlock.model_construct(
                        id=A,
                        created_at=A,
                        thinking="Let me think...",
                    ),
                    TextBlock.model_construct(
                        id=A,
                        created_at=A,
                        text="Answer",
                    ),
                ],
            ),
        )


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestOllamaStream(IsolatedAsyncioTestCase):
    """Tests for OllamaChatModel in streaming mode."""

    def setUp(self) -> None:
        self.model = _make_model(stream=True)
        # Client is built eagerly in __init__; inject a mock onto the
        # instance so chat() hits it instead of the network.
        self.mock_client = MagicMock()
        self.model.client = self.mock_client

    async def test_stream_text(self) -> None:
        """Stream text yields deltas then full content."""
        chunks = [
            _make_stream_chunk(content="Hi"),
            _make_stream_chunk(content=" there"),
        ]
        self.mock_client.chat = AsyncMock(
            return_value=_MockAsyncStream(chunks),
        )

        gen = await self.model([])
        responses = [r async for r in gen]

        self.assertListEqual(
            [(r.is_last, r.content) for r in responses],
            [
                (
                    False,
                    [TextBlock.model_construct(id=A, created_at=A, text="Hi")],
                ),
                (
                    False,
                    [
                        TextBlock.model_construct(
                            id=A,
                            created_at=A,
                            text=" there",
                        ),
                    ],
                ),
                (
                    True,
                    [
                        TextBlock.model_construct(
                            id=A,
                            created_at=A,
                            text="Hi there",
                        ),
                    ],
                ),
            ],
        )

    async def test_stream_thinking_and_text(self) -> None:
        """Stream thinking and text deltas then final with accumulated
        content."""
        chunks = [
            _make_stream_chunk(thinking="Think step"),
            _make_stream_chunk(content="Result"),
        ]
        self.mock_client.chat = AsyncMock(
            return_value=_MockAsyncStream(chunks),
        )

        gen = await self.model([])
        responses = [r async for r in gen]

        self.assertListEqual(
            [(r.is_last, r.content) for r in responses],
            [
                (
                    False,
                    [
                        ThinkingBlock.model_construct(
                            id=A,
                            created_at=A,
                            thinking="Think step",
                        ),
                    ],
                ),
                (
                    False,
                    [
                        TextBlock.model_construct(
                            id=A,
                            created_at=A,
                            text="Result",
                        ),
                    ],
                ),
                (
                    True,
                    [
                        ThinkingBlock.model_construct(
                            id=A,
                            created_at=A,
                            thinking="Think step",
                        ),
                        TextBlock.model_construct(
                            id=A,
                            created_at=A,
                            text="Result",
                        ),
                    ],
                ),
            ],
        )

    async def test_stream_tool_call(self) -> None:
        """Stream tool-call chunk yields delta then final with same
        ToolCallBlock."""
        chunks = [
            _make_stream_chunk(
                tool_calls=[
                    {"name": "search", "args": {"q": "hello"}},
                ],
            ),
        ]
        self.mock_client.chat = AsyncMock(
            return_value=_MockAsyncStream(chunks),
        )

        gen = await self.model([])
        responses = [r async for r in gen]

        tool_block = ToolCallBlock.model_construct(
            id=A,
            created_at=A,
            name="search",
            input=json.dumps({"q": "hello"}),
        )
        self.assertListEqual(
            [(r.is_last, r.content) for r in responses],
            [
                (False, [tool_block]),
                (True, [tool_block]),
            ],
        )


# ---------------------------------------------------------------------------
# Tool-call identity regressions
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    importlib.util.find_spec("ollama"),
    "ollama not installed",
)
class TestOllamaToolCallIdentities(IsolatedAsyncioTestCase):
    """Tool identities must survive streaming and distinguish agent rounds."""

    async def test_non_stream_agent_calls_same_tool_twice(self) -> None:
        """Both non-streaming tool rounds execute within one agent reply."""
        await self._check_agent_tool_rounds(stream=False)

    async def test_stream_agent_calls_same_tool_twice(self) -> None:
        """Both streaming tool rounds execute within one agent reply."""
        await self._check_agent_tool_rounds(stream=True)

    async def _check_agent_tool_rounds(self, stream: bool) -> None:
        """Run two tool rounds and check their results in the agent state.

        Args:
            stream (`bool`):
                Whether the mocked Ollama SDK returns streaming responses.
        """
        executed = []

        async def lookup(value: str) -> ToolChunk:
            """Return the requested value.

            Args:
                value (`str`):
                    The value to look up.

            Returns:
                `ToolChunk`:
                    The lookup result.
            """
            executed.append(value)
            return ToolChunk(content=[TextBlock(text=value)])

        responses = [
            _mock_completion(
                tool_calls=[{"name": "lookup", "args": {"value": value}}],
            )
            for value in ("first", "second")
        ] + [_mock_completion(content="done")]
        model = _make_model(stream=stream)
        model.client.chat = AsyncMock(
            side_effect=(
                [_MockAsyncStream([r]) for r in responses]
                if stream
                else responses
            ),
        )
        agent = Agent(
            name="test",
            system_prompt="Use lookup twice.",
            model=model,
            toolkit=Toolkit(
                tools=[
                    FunctionTool(
                        lookup,
                        permission=PermissionDecision(
                            behavior=PermissionBehavior.ALLOW,
                            message="Allow the test lookup tool.",
                        ),
                    ),
                ],
            ),
            injection_config=InjectionConfig(inject_runtime_state=False),
            react_config=ReActConfig(max_iters=4),
        )

        result = await agent.reply(
            UserMsg(name="user", content="Look up first and second."),
        )

        self.assertEqual(executed, ["first", "second"])
        self.assertEqual(model.client.chat.await_count, 3)
        calls = agent.state.context[-1].get_content_blocks("tool_call")
        self.assertEqual(len({call.id for call in calls}), 2)
        expected_blocks = []
        for call, value in zip(calls, ("first", "second")):
            expected_blocks.extend(
                [
                    ToolCallBlock.model_construct(
                        id=call.id,
                        created_at=A,
                        name="lookup",
                        input=json.dumps({"value": value}),
                        state="finished",
                    ),
                    ToolResultBlock.model_construct(
                        id=call.id,
                        created_at=A,
                        name="lookup",
                        output=[
                            TextBlock.model_construct(
                                id=A,
                                created_at=A,
                                text=value,
                            ),
                        ],
                        state="success",
                    ),
                ],
            )
        expected_blocks.append(
            TextBlock.model_construct(id=A, created_at=A, text="done"),
        )
        self.assertEqual(agent.state.context[-1].content, expected_blocks)
        self.assertEqual(result.content, [expected_blocks[-1]])
        self.assertEqual(result.finished_reason, ReplyFinishedReason.COMPLETED)
        self.assertEqual(agent.state.get_unfinished_tool_calls(agent.name), [])

    async def test_non_stream_multiple_tool_ids(self) -> None:
        """Same-name calls in and across completions have distinct IDs."""
        model = _make_model()
        tool_calls = [
            {"name": "lookup", "args": {"value": value}}
            for value in ("first", "second")
        ]
        model.client.chat = AsyncMock(
            side_effect=[
                _mock_completion(tool_calls=tool_calls) for _ in range(2)
            ],
        )

        responses = [await model([]) for _ in range(2)]

        self.assertEqual(
            [response.content for response in responses],
            [
                [
                    ToolCallBlock.model_construct(
                        id=A,
                        created_at=A,
                        name=call["name"],
                        input=json.dumps(call["args"]),
                    )
                    for call in tool_calls
                ],
            ]
            * 2,
        )
        self.assertEqual(
            len({block.id for r in responses for block in r.content}),
            4,
        )

    async def test_stream_tool_ids_stable_within_response(self) -> None:
        """Repeated stream entries reuse IDs only within the same response."""
        model = _make_model(stream=True)
        tool_calls = [
            {"name": "lookup", "args": {"value": "first"}},
            {"name": "lookup", "args": {"value": "second"}},
        ]
        responses = []
        for _ in range(2):
            chunks = _MockAsyncStream(
                [_make_stream_chunk(tool_calls=tool_calls) for _ in range(2)],
            )
            responses.append(
                [
                    r
                    async for r in model._parse_stream_response(
                        datetime.now(),
                        chunks,
                    )
                ],
            )

        for response in responses:
            expected = [
                ToolCallBlock.model_construct(
                    id=block.id,
                    created_at=A,
                    name=call["name"],
                    input=json.dumps(call["args"]),
                )
                for block, call in zip(response[0].content, tool_calls)
            ]
            self.assertEqual([r.content for r in response], [expected] * 2)
        self.assertEqual(
            len({b.id for response in responses for b in response[0].content}),
            4,
        )


# ---------------------------------------------------------------------------
# _format_tools tests
# ---------------------------------------------------------------------------

_FT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the time",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"],
            },
        },
    },
]


class TestOllamaFormatTools(unittest.TestCase):
    """Tests for OllamaChatModel._format_tools."""

    def setUp(self) -> None:
        self.model = _make_model()

    def test_tools_forwarded_no_choice(self) -> None:
        """Tools are forwarded unchanged when tool_choice is None."""
        fmt_tools, fmt_choice = self.model._format_tools(_FT_TOOLS, None)
        self.assertEqual(fmt_tools, _FT_TOOLS)
        self.assertIsNone(fmt_choice)

    def test_tools_filtered(self) -> None:
        """ToolChoice with tools list filters to matching function names."""
        fmt_tools, fmt_choice = self.model._format_tools(
            _FT_TOOLS,
            ToolChoice(mode="auto", tools=["get_weather"]),
        )
        self.assertIsNotNone(fmt_tools)
        assert fmt_tools is not None
        self.assertEqual(len(fmt_tools), 1)
        self.assertEqual(fmt_tools[0]["function"]["name"], "get_weather")
        self.assertIsNone(fmt_choice)
