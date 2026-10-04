"""Application composition root: wire the concrete components together.

This is the one module allowed to know the concrete implementations. It builds
(optionally) an LLM client, the workers, the registry, the planner, the
evaluator and the aggregator, injects them into an ``Orchestrator`` and returns
it. It only constructs and injects: it never plans, routes, executes, evaluates,
aggregates or calls a model itself.
"""

from __future__ import annotations

from .aggregators import LLMAggregator
from .evaluators import LLMEvaluator
from .llm import LLMClient, build_llm_client
from .orchestrator import Orchestrator
from .planner import LLMPlanner
from .workers import CodingWorker, ResearchWorker, WorkerRegistry


def build_orchestrator(llm: LLMClient | None = None) -> Orchestrator:
    """Build an :class:`Orchestrator` with every component wired up.

    The same ``llm`` instance is shared by the planner, both workers, the
    evaluator and the aggregator, so swapping providers is a matter of passing a
    different client. When ``llm`` is ``None`` the client is built from the
    configured provider via
    :func:`orchestrator_worker.llm.build_llm_client`; callers and tests that
    must stay offline should pass a client explicitly.

    Building performs no work: no plan is made, no worker runs and the model is
    never called.
    """
    client = llm if llm is not None else build_llm_client()

    registry = WorkerRegistry()
    registry.register(ResearchWorker(client))
    registry.register(CodingWorker(client))

    planner = LLMPlanner(client, available_workers=registry.names())
    evaluator = LLMEvaluator(client)
    aggregator = LLMAggregator(client)

    return Orchestrator(
        registry,
        planner=planner,
        evaluator=evaluator,
        aggregator=aggregator,
    )