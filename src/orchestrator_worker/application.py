"""Application composition root: wire the concrete components together.

This is the one module allowed to know the concrete implementations. It builds
(optionally) an LLM client, the built-in tools, the tool runtime, the workers,
the registry, the planner, the evaluator and the aggregator, injects them into
an ``Orchestrator`` and returns it. It only constructs and injects: it never
plans, routes, executes, evaluates, aggregates or calls a model itself.

The ``"research"`` worker is tool-enabled: it receives a shared ``ToolLoop``
plus its own allow-list of ``Tool`` definitions. The tool-free ``"coding"``
worker uses the same client directly.

The planner is capability-aware: it receives ``registry.capabilities()``, so
the prompt lists each registered worker together with the capability the worker
itself declares. Registering another worker is enough to make its capability
visible to the planner - nothing else has to be updated.
"""

from __future__ import annotations

from .aggregators import LLMAggregator
from .builtin_tools import calculator_tool
from .evaluators import LLMEvaluator
from .llm import LLMClient, build_llm_client
from .orchestrator import Orchestrator
from .planner import LLMPlanner
from .tool_loop import ToolLoop
from .tools import ToolExecutor, ToolRegistry
from .workers import CodingWorker, WorkerRegistry
from .workers.research_tool import ResearchToolWorker


def build_orchestrator(llm: LLMClient | None = None) -> Orchestrator:
    """Build an :class:`Orchestrator` with every component wired up.

    The same ``llm`` instance reaches the planner, the research worker (through
    its tool loop), the coding worker, the evaluator and the aggregator, so
    swapping providers is a matter of passing a different client. When ``llm``
    is ``None`` the client is built from the configured provider via
    :func:`orchestrator_worker.llm.build_llm_client`; callers and tests that
    must stay offline should pass a client explicitly.

    The built-in calculator is registered in a :class:`ToolRegistry`, executed
    through a :class:`ToolExecutor` and offered to the research worker through a
    single shared :class:`ToolLoop`. The worker gets its own allow-list, never
    the registry.

    The planner is built from ``registry.capabilities()`` rather than from bare
    worker names, so its prompt shows the model what each worker can do.

    Building performs no work: no plan is made, no worker runs and the model is
    never called.
    """
    client = llm if llm is not None else build_llm_client()

    calculator = calculator_tool()
    tool_registry = ToolRegistry()
    tool_registry.register(calculator)
    tool_loop = ToolLoop(client, ToolExecutor(tool_registry))

    registry = WorkerRegistry()
    registry.register(ResearchToolWorker(tool_loop, tools=(calculator,)))
    registry.register(CodingWorker(client))

    planner = LLMPlanner(client, available_workers=registry.capabilities())
    evaluator = LLMEvaluator(client)
    aggregator = LLMAggregator(client)

    return Orchestrator(
        registry,
        planner=planner,
        evaluator=evaluator,
        aggregator=aggregator,
    )