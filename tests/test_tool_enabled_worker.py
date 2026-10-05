"""Phase 5C-4A tests: the tool-enabled worker boundary.

Everything here is offline. The worker is driven by a recording ``ToolLoop``
double, so no model call and no tool execution happens.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from orchestrator_worker.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
)
from orchestrator_worker.tool_loop import ToolLoop, ToolLoopError
from orchestrator_worker.tools import (
    Tool,
    ToolCall,
    ToolExecutionError,
    ToolExecutor,
    ToolRegistry,
    ToolRegistryError,
)
from orchestrator_worker.workers.base import BaseWorker, WorkerError
from orchestrator_worker.workers.tool_enabled import ToolEnabledWorker

MODULE_NAME = "orchestrator_worker.workers.tool_enabled"


class StubLLMClient(LLMClient):
    """LLMClient that the worker must never reach: the loop owns model calls."""

    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        raise AssertionError(
            "the worker must run through the ToolLoop, not the LLMClient"
        )


class RecordingToolLoop(ToolLoop):
    """ToolLoop double: records requests and returns a canned response."""

    def __init__(
        self,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(StubLLMClient(), ToolExecutor(ToolRegistry()))
        self._response = (
            response
            if response is not None
            else LLMResponse(content="worker answer", model="fake-model")
        )
        self._error = error
        self.requests: list[LLMRequest] = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def run(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response


class ExampleToolWorker(ToolEnabledWorker):
    """Minimal subclass: it only supplies identity and prompt."""

    name = "example"
    system_prompt = "You are an example tool-enabled worker."


def make_tool(name: str = "alpha") -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        parameters={},
        handler=lambda args: "ok",
    )


def make_worker(loop=None, tools=None, **kwargs) -> ExampleToolWorker:
    return ExampleToolWorker(
        loop if loop is not None else RecordingToolLoop(),
        tools=tools if tools is not None else (make_tool(),),
        **kwargs,
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


def _imports_of(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    return _imports_of_path(pathlib.Path(module.__file__))


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


def test_tool_enabled_worker_is_a_base_worker():
    assert issubclass(ToolEnabledWorker, BaseWorker)
    assert isinstance(make_worker(), BaseWorker)


def test_default_name_is_overridden_by_the_subclass():
    assert ExampleToolWorker.name == "example"
    assert make_worker().name == "example"


# -------------------------------------------------------------- construction


def test_construction_requires_a_tool_loop():
    with pytest.raises(WorkerError, match="expects a ToolLoop"):
        ExampleToolWorker(None, tools=(make_tool(),))


@pytest.mark.parametrize("bad", [object(), None, "loop", 42, LLMClient])
def test_rejects_a_non_tool_loop(bad):
    with pytest.raises(WorkerError, match="expects a ToolLoop"):
        ExampleToolWorker(bad, tools=(make_tool(),))


def test_construction_requires_tools():
    with pytest.raises(WorkerError, match="tools must be a tuple of Tool"):
        ExampleToolWorker(RecordingToolLoop())


@pytest.mark.parametrize("bad", [[], [make_tool()], "alpha", 42, None, iter(())])
def test_rejects_non_tuple_tools(bad):
    with pytest.raises(WorkerError, match="tools must be a tuple of Tool"):
        ExampleToolWorker(RecordingToolLoop(), tools=bad)


@pytest.mark.parametrize("bad", ["alpha", 42, None, object()])
def test_rejects_non_tool_members(bad):
    with pytest.raises(WorkerError, match=r"tools\[0\] must be a Tool"):
        ExampleToolWorker(RecordingToolLoop(), tools=(bad,))


def test_reports_the_index_of_a_bad_member():
    with pytest.raises(WorkerError, match=r"tools\[1\] must be a Tool"):
        ExampleToolWorker(RecordingToolLoop(), tools=(make_tool(), "nope"))


def test_accepts_an_empty_tools_tuple():
    worker = ExampleToolWorker(RecordingToolLoop(), tools=())

    assert isinstance(worker, BaseWorker)


def test_uses_the_injected_tool_loop_instance():
    loop = RecordingToolLoop(LLMResponse(content="answer", model="fake-model"))
    unused = RecordingToolLoop()

    worker = ExampleToolWorker(loop, tools=(make_tool(),))

    assert worker.execute("do it") == "answer"
    assert loop.calls == 1
    assert unused.calls == 0

# ----------------------------------------------------- request construction


def test_execute_runs_the_tool_loop_once():
    loop = RecordingToolLoop()

    make_worker(loop).execute("do it")

    assert loop.calls == 1


def test_run_receives_an_llm_request():
    loop = RecordingToolLoop()

    make_worker(loop).execute("do it")

    assert type(loop.requests[0]) is LLMRequest


def test_request_tools_are_the_worker_tools():
    tools = (make_tool("beta"), make_tool("alpha"))
    loop = RecordingToolLoop()

    ExampleToolWorker(loop, tools=tools).execute("do it")

    assert loop.requests[0].tools is tools


def test_request_tools_keep_their_order():
    tools = (make_tool("beta"), make_tool("alpha"))
    loop = RecordingToolLoop()

    ExampleToolWorker(loop, tools=tools).execute("do it")

    assert [tool.name for tool in loop.requests[0].tools] == ["beta", "alpha"]


def test_worker_needs_no_tool_registry():
    lonely = make_tool("never-registered")
    loop = RecordingToolLoop()

    ExampleToolWorker(loop, tools=(lonely,)).execute("do it")

    assert loop.requests[0].tools == (lonely,)


def test_request_carries_a_system_and_a_user_message():
    loop = RecordingToolLoop()

    make_worker(loop).execute("summarise the release notes")

    messages = loop.requests[0].messages
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == ExampleToolWorker.system_prompt
    assert messages[1].content == "summarise the release notes"


def test_subclass_system_prompt_is_used():
    class PromptedWorker(ToolEnabledWorker):
        name = "prompted"
        system_prompt = "a custom system prompt"

    loop = RecordingToolLoop()

    PromptedWorker(loop, tools=(make_tool(),)).execute("task")

    assert loop.requests[0].messages[0].content == "a custom system prompt"


def test_build_request_is_the_construction_seam():
    class OverridingWorker(ToolEnabledWorker):
        name = "overriding"

        def build_request(self, task: str) -> LLMRequest:
            return LLMRequest(
                messages=[Message.user(f"custom:{task}")],
                tools=self._tools,
            )

    loop = RecordingToolLoop()

    OverridingWorker(loop, tools=(make_tool(),)).execute("task")

    assert [m.content for m in loop.requests[0].messages] == ["custom:task"]


def test_task_is_stripped_before_it_reaches_the_request():
    loop = RecordingToolLoop()

    make_worker(loop).execute("   spaced task   ")

    assert loop.requests[0].messages[1].content == "spaced task"


def test_request_keeps_worker_llm_options():
    loop = RecordingToolLoop()

    make_worker(loop, temperature=0.3, max_tokens=128).execute("do it")

    request = loop.requests[0]
    assert request.temperature == 0.3
    assert request.max_tokens == 128
    assert request.json_mode is False


# ------------------------------------------------------------------- result


def test_returns_the_response_content_as_a_string():
    loop = RecordingToolLoop(LLMResponse(content="final answer", model="fake-model"))

    result = make_worker(loop).execute("do it")

    assert result == "final answer"
    assert isinstance(result, str)


def test_result_content_is_stripped():
    loop = RecordingToolLoop(LLMResponse(content="  padded  ", model="fake-model"))

    assert make_worker(loop).execute("do it") == "padded"


def test_tool_calls_do_not_leak_into_the_result():
    call = ToolCall(id="call-1", name="alpha", arguments={})
    loop = RecordingToolLoop(
        LLMResponse(content="done", model="fake-model", tool_calls=(call,))
    )

    result = make_worker(loop).execute("do it")

    assert result == "done"
    assert isinstance(result, str)


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_response_content_is_rejected(content):
    loop = RecordingToolLoop(LLMResponse(content=content, model="fake-model"))

    with pytest.raises(WorkerError, match="empty response"):
        make_worker(loop).execute("do it")


@pytest.mark.parametrize("task", ["", "   ", None])
def test_empty_task_is_rejected_before_the_loop_runs(task):
    loop = RecordingToolLoop()

    with pytest.raises(WorkerError, match="empty task"):
        make_worker(loop).execute(task)

    assert loop.calls == 0


# ------------------------------------------------------- configuration reuse


def test_multiple_executions_share_one_configuration():
    tools = (make_tool(),)
    loop = RecordingToolLoop()
    worker = ExampleToolWorker(loop, tools=tools)

    worker.execute("first")
    worker.execute("second")

    assert loop.calls == 2
    assert loop.requests[0].tools is tools
    assert loop.requests[1].tools is tools
    assert loop.requests[1].messages[1].content == "second"


def test_execute_does_not_modify_its_tools_or_task():
    tools = (make_tool("beta"), make_tool("alpha"))
    task = "keep me"
    loop = RecordingToolLoop()
    worker = ExampleToolWorker(loop, tools=tools)
    tools_before = tuple(tools)

    worker.execute(task)

    assert tools == tools_before
    assert [tool.name for tool in tools] == ["beta", "alpha"]
    assert task == "keep me"

# ------------------------------------------------------------------- errors


def test_worker_error_is_a_runtime_error():
    assert issubclass(WorkerError, RuntimeError)


def test_tool_loop_error_becomes_a_worker_error():
    error = ToolLoopError("tool loop exceeded max_rounds=8")
    loop = RecordingToolLoop(error=error)

    with pytest.raises(WorkerError, match="example worker tool loop failed") as info:
        make_worker(loop).execute("do it")

    assert type(info.value) is WorkerError
    assert info.value.__cause__ is error


def test_tool_registry_error_becomes_a_worker_error():
    error = ToolRegistryError("unknown tool 'missing'")
    loop = RecordingToolLoop(error=error)

    with pytest.raises(WorkerError, match="example worker tool call failed") as info:
        make_worker(loop).execute("do it")

    assert info.value.__cause__ is error


def test_llm_error_becomes_a_worker_error():
    error = LLMError("provider down")
    loop = RecordingToolLoop(error=error)

    with pytest.raises(WorkerError, match="example worker LLM call failed") as info:
        make_worker(loop).execute("do it")

    assert info.value.__cause__ is error


def test_tool_execution_error_is_left_to_the_tool_loop():
    error = ToolExecutionError("tool 'alpha' failed while executing")
    loop = RecordingToolLoop(error=error)

    with pytest.raises(ToolExecutionError) as info:
        make_worker(loop).execute("do it")

    assert info.value is error


def test_worker_never_runs_tool_calls_after_a_loop_failure():
    loop = RecordingToolLoop(error=ToolLoopError("boom"))

    with pytest.raises(WorkerError):
        make_worker(loop).execute("do it")

    assert loop.calls == 1


# ----------------------------------------------------------------- isolation


def test_module_imports_only_the_allowed_names():
    assert _imports_of(MODULE_NAME) == [
        "__future__",
        "..llm",
        "..tool_loop",
        "..tools",
        ".base",
    ]


@pytest.mark.parametrize(
    "token",
    [
        "openai",
        "deepseek",
        "anthropic",
        "DeepSeekClient",
        "WorkerRegistry",
        "ToolRegistry",
        "ResearchWorker",
        "CodingWorker",
    ],
)
def test_module_never_imports_a_forbidden_name(token):
    assert not any(token in name for name in _imports_of(MODULE_NAME))


@pytest.mark.parametrize(
    "identifier",
    [
        "complete",
        "handler",
        "max_retries",
        "replan",
        "evaluate",
        "aggregate",
        "registry",
        "ToolRegistry",
        "ToolExecutor",
        "ResearchWorker",
        "CodingWorker",
        "DeepSeekClient",
    ],
)
def test_module_never_references_a_forbidden_identifier(identifier):
    assert identifier not in module_identifiers()


def test_module_uses_the_tool_loop_as_its_only_runtime_entry():
    source = module_source()
    identifiers = module_identifiers()

    assert "run" in identifiers
    assert source.count("self._tool_loop.run(") == 1


def test_the_worker_tests_never_import_a_provider_sdk():
    imported = _imports_of_path(pathlib.Path(__file__))

    assert not any(
        token in name
        for name in imported
        for token in ("openai", "deepseek", "anthropic")
    )


# ---------------------------------------------------------------- regression


def test_existing_workers_stay_separate_and_untouched():
    from orchestrator_worker.workers import CodingWorker, ResearchWorker

    assert not issubclass(ResearchWorker, ToolEnabledWorker)
    assert not issubclass(CodingWorker, ToolEnabledWorker)
    assert ResearchWorker.name == "research"
    assert CodingWorker.name == "coding"