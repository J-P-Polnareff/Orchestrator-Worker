"""Phase 5C-4B tests: ResearchToolWorker.

Offline only. The tool-loop-level tests use a recording ``ToolLoop`` double; the
integration test uses the real ``ToolLoop`` + ``ToolExecutor`` + calculator with
a scripted ``LLMClient``, so no model call leaves the process.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from orchestrator_worker.builtin_tools import calculator_tool
from orchestrator_worker.llm import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
)
from orchestrator_worker.tool_loop import ToolLoop
from orchestrator_worker.tools import Tool, ToolCall, ToolExecutor, ToolRegistry
from orchestrator_worker.workers.base import BaseWorker, WorkerError
from orchestrator_worker.workers.research import ResearchWorker
from orchestrator_worker.workers.research_tool import (
    DEFAULT_SYSTEM_PROMPT,
    ResearchToolWorker,
)
from orchestrator_worker.workers.tool_enabled import ToolEnabledWorker

MODULE_NAME = "orchestrator_worker.workers.research_tool"


class StubLLMClient(LLMClient):
    """LLMClient the worker must never reach directly."""

    name = "stub"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("the worker must go through the ToolLoop")


class ScriptedLLMClient(LLMClient):
    """Returns scripted responses in order and records every request."""

    name = "scripted"

    def __init__(self, *script: LLMResponse) -> None:
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self.script):
            raise AssertionError(
                f"unexpected LLM call #{index + 1}; the script has "
                f"{len(self.script)} entries."
            )
        return self.script[index]


class RecordingToolLoop(ToolLoop):
    """ToolLoop double: records requests and returns a canned response."""

    def __init__(self, response: LLMResponse | None = None) -> None:
        super().__init__(StubLLMClient(), ToolExecutor(ToolRegistry()))
        self._response = (
            response
            if response is not None
            else LLMResponse(content="worker answer", model="fake-model")
        )
        self.requests: list[LLMRequest] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def run(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._response


def make_worker(loop=None, tools=None, **kwargs) -> ResearchToolWorker:
    return ResearchToolWorker(
        loop if loop is not None else RecordingToolLoop(),
        tools=tools if tools is not None else (calculator_tool(),),
        **kwargs,
    )


def spying_calculator() -> tuple[Tool, list[dict]]:
    """Return the calculator with a recording handler.

    Mirrors the built-in definition but wraps the handler so a test can prove
    the tool was really executed with the parsed arguments.
    """
    real = calculator_tool()
    seen: list[dict] = []

    def spy(arguments):
        seen.append(dict(arguments))
        return real.handler(arguments)

    return (
        Tool(
            name=real.name,
            description=real.description,
            parameters=real.parameters,
            handler=spy,
        ),
        seen,
    )


def module_source() -> str:
    module = importlib.import_module(MODULE_NAME)
    return pathlib.Path(module.__file__).read_text(encoding="utf-8")


def _imports_of_path(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    return imported


def module_identifiers() -> set[str]:
    identifiers: set[str] = set()
    for node in ast.walk(ast.parse(module_source())):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            identifiers.update(alias.asname or alias.name for alias in node.names)
    return identifiers

# ------------------------------------------------------------------ identity


def test_research_tool_worker_is_a_tool_enabled_worker():
    assert issubclass(ResearchToolWorker, ToolEnabledWorker)
    assert issubclass(ResearchToolWorker, BaseWorker)
    assert isinstance(make_worker(), BaseWorker)


def test_worker_name_matches_the_planner_contract():
    assert ResearchToolWorker.name == "research"
    assert ResearchToolWorker.name == ResearchWorker.name


def test_default_system_prompt_is_research_oriented():
    assert ResearchToolWorker.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert DEFAULT_SYSTEM_PROMPT.startswith("You are a research worker.")
    assert "calculator" in DEFAULT_SYSTEM_PROMPT
    assert "tool" in DEFAULT_SYSTEM_PROMPT


# -------------------------------------------------------------- construction


def test_tools_must_be_injected():
    with pytest.raises(WorkerError, match="tools must be a tuple of Tool"):
        ResearchToolWorker(RecordingToolLoop())


@pytest.mark.parametrize("bad", [[], [calculator_tool()], "calculator", None])
def test_rejects_non_tuple_tools(bad):
    with pytest.raises(WorkerError, match="tools must be a tuple of Tool"):
        ResearchToolWorker(RecordingToolLoop(), tools=bad)


def test_rejects_a_non_tool_member():
    with pytest.raises(WorkerError, match=r"tools\[0\] must be a Tool"):
        ResearchToolWorker(RecordingToolLoop(), tools=("calculator",))


def test_keeps_the_injected_tools_tuple():
    tools = (calculator_tool(),)
    worker = ResearchToolWorker(RecordingToolLoop(), tools=tools)

    assert worker._tools is tools


# ------------------------------------------------------------ build_request


def test_build_request_returns_an_llm_request():
    request = make_worker().build_request("compute 17 times 23")

    assert type(request) is LLMRequest


def test_build_request_carries_the_system_prompt_and_task():
    request = make_worker().build_request("compute 17 times 23")

    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0] == Message.system(DEFAULT_SYSTEM_PROMPT)
    assert request.messages[1] == Message.user("compute 17 times 23")


def test_build_request_carries_the_allowed_tools():
    tools = (calculator_tool(),)
    worker = ResearchToolWorker(RecordingToolLoop(), tools=tools)

    request = worker.build_request("compute 17 times 23")

    assert request.tools is tools


def test_build_request_carries_the_worker_llm_options():
    request = make_worker(temperature=0.1, max_tokens=64).build_request("task")

    assert request.temperature == 0.1
    assert request.max_tokens == 64


# ------------------------------------------------------------------ execute


def test_execute_calls_the_tool_loop_once():
    loop = RecordingToolLoop()

    make_worker(loop).execute("compute 17 times 23")

    assert loop.calls == 1


def test_execute_returns_the_response_content():
    loop = RecordingToolLoop(
        LLMResponse(content=" 17 x 23 = 391. ", model="fake-model")
    )

    result = make_worker(loop).execute("compute 17 times 23")

    assert result == "17 x 23 = 391."
    assert isinstance(result, str)


def test_execute_sends_the_allowed_tools_to_the_loop():
    tools = (calculator_tool(),)
    loop = RecordingToolLoop()
    worker = ResearchToolWorker(loop, tools=tools)

    worker.execute("compute 17 times 23")

    assert loop.requests[0].tools is tools


@pytest.mark.parametrize("task", ["", "   ", None])
def test_empty_task_is_rejected_before_the_loop_runs(task):
    loop = RecordingToolLoop()

    with pytest.raises(WorkerError, match="empty task"):
        make_worker(loop).execute(task)

    assert loop.calls == 0


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_response_is_rejected(content):
    loop = RecordingToolLoop(LLMResponse(content=content, model="fake-model"))

    with pytest.raises(WorkerError, match="empty response"):
        make_worker(loop).execute("task")


# ------------------------------------------------- real tool loop integration


def build_real_worker():
    """Wire the real ToolLoop, ToolExecutor and calculator together."""
    tool, seen = spying_calculator()
    registry = ToolRegistry()
    registry.register(tool)
    llm = ScriptedLLMClient(
        LLMResponse(
            content="",
            model="fake-model",
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="calculator",
                    arguments={"a": 17, "b": 23, "operation": "multiply"},
                ),
            ),
        ),
        LLMResponse(content="17 x 23 = 391.", model="fake-model"),
    )
    loop = ToolLoop(llm, ToolExecutor(registry))
    worker = ResearchToolWorker(loop, tools=(tool,))
    return worker, llm, tool, seen


def test_real_tool_loop_executes_the_calculator():
    worker, llm, _tool, seen = build_real_worker()

    result = worker.execute("compute 17 times 23")

    assert result == "17 x 23 = 391."
    assert seen == [{"a": 17, "b": 23, "operation": "multiply"}]
    assert len(llm.requests) == 2


def test_real_tool_loop_history_is_assistant_then_tool():
    worker, llm, tool, _seen = build_real_worker()

    worker.execute("compute 17 times 23")

    follow_up = llm.requests[1]
    assert [message.role for message in follow_up.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert follow_up.messages[2].tool_calls == (
        ToolCall(
            id="call-1",
            name="calculator",
            arguments={"a": 17, "b": 23, "operation": "multiply"},
        ),
    )
    assert follow_up.messages[3].tool_call_id == "call-1"
    assert follow_up.messages[3].content == "391"
    assert follow_up.tools == (tool,)


# ----------------------------------------------------------------- isolation


def test_module_imports_only_the_tool_enabled_worker():
    module = importlib.import_module(MODULE_NAME)

    assert _imports_of_path(pathlib.Path(module.__file__)) == [
        "__future__",
        ".tool_enabled",
    ]


@pytest.mark.parametrize(
    "token",
    [
        "openai",
        "deepseek",
        "anthropic",
        "DeepSeekClient",
        "ToolRegistry",
        "WorkerRegistry",
        "CodingWorker",
        "ResearchWorker",
    ],
)
def test_module_never_imports_a_forbidden_name(token):
    module = importlib.import_module(MODULE_NAME)
    imported = _imports_of_path(pathlib.Path(module.__file__))

    assert not any(token in name for name in imported)


@pytest.mark.parametrize(
    "identifier",
    [
        "registry",
        "max_retries",
        "replan",
        "evaluate",
        "aggregate",
        "complete",
        "handler",
        "ToolRegistry",
        "ToolExecutor",
        "WorkerRegistry",
    ],
)
def test_module_never_references_a_forbidden_identifier(identifier):
    assert identifier not in module_identifiers()


def test_the_research_tool_tests_never_import_a_provider_sdk():
    imported = _imports_of_path(pathlib.Path(__file__))

    assert not any(
        token in name
        for name in imported
        for token in ("openai", "deepseek", "anthropic")
    )


def test_existing_research_worker_is_untouched_and_separate():
    assert not issubclass(ResearchWorker, ToolEnabledWorker)
    assert ResearchWorker.name == "research"