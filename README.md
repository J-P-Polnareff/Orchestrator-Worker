# Orchestrator-Worker

A multi-agent backend prototype. Target request flow:

```
User -> Orchestrator -> Planner -> Worker Router -> Workers
     -> Evaluator -> Aggregator -> Final Answer
```

Current status: **Phase 4E (second step)**. Evaluation and evaluator-driven
whole-plan retry are implemented; the request flow below is still unchanged.

Routing - `Orchestrator.run(task, worker_name=...)`:

```
User Task -> Orchestrator -> WorkerRegistry -> ResearchWorker / CodingWorker
          -> Worker Result -> Final Result
```

Planning - `Orchestrator.plan(task)`:

```
User Task -> Orchestrator -> Planner -> Plan        (stops here)
```

Plan execution - `Orchestrator.execute_plan(plan)`:

```
Plan -> Router -> Worker (per step) -> AgentState -> ExecutionResult[]
     -> ExecutionContext
```

Aggregation - `Orchestrator.aggregate(context)`:

```
ExecutionContext -> Aggregator -> Final Answer
```

Evaluation (standalone) - `LLMEvaluator.evaluate(context)`:

```
ExecutionContext -> Evaluator -> EvaluationResult
```

Retry - `Orchestrator.execute_plan_with_retry(plan, max_retries=...)`:

```
Plan -> execute -> evaluate -> passed? -> ExecutionContext
                            -> failed? -> execute the whole plan again
                            -> budget spent? -> RetryError
```

`plan()` only produces a `Plan`: the planner never executes a step and has no
access to the worker registry. `execute_plan()` then walks `plan.steps` in
order, resolving each `step.worker_name` through a deterministic router (no
LLM involved) and calling that worker once per step. Each result is an
`ExecutionResult` that carries the `step_id`, `task` and `worker_name` of the
step it came from next to the `AgentState` the worker produced, so a result
never has to be matched back to its step by list position.

In short: a `Plan` says what to do, one `ExecutionResult` records one executed
step, and an `ExecutionContext` is the complete result set of one plan.
`execute_plan()` still returns the plain result list; `execute_plan_context()`
returns the same results wrapped in an `ExecutionContext`, and `aggregate()`
turns that context into the final answer. Execution is strictly sequential and
aggregation only runs once every step has finished. Evaluation exists as a
standalone component (`Evaluator` / `LLMEvaluator`), and
`execute_plan_with_retry()` re-runs the whole plan when the evaluator reports
`passed=False`. There is no replanning, no targeted per-step retry, no
automatic aggregation and no parallel execution.

## Design constraints

- Python, backend only, no UI in this phase.
- DeepSeek is the only live LLM today, but every model call goes through
  `orchestrator_worker.llm.base.LLMClient`. Adding a Claude adapter means
  writing one class and registering it; agent code does not change.
- Tests must run with no network access and no API key.

## Layout

```
src/orchestrator_worker/
  config.py        env / .env -> immutable Settings
  state.py         AgentState shared across the pipeline
  plan.py          Plan / PlanStep data model
  execution.py     ExecutionResult: a plan step plus the state it produced
  context.py       ExecutionContext: a plan plus its complete result set
  aggregators/
    base.py        Aggregator contract + AggregatorError
    llm.py         LLMAggregator: ExecutionContext -> final answer
  evaluation.py    EvaluationResult + EvaluationError
  evaluators/
    base.py        Evaluator contract + EvaluatorError
    llm.py         LLMEvaluator: ExecutionContext -> EvaluationResult
  retry.py         RetryError: evaluation failed, retry budget exhausted
  router.py        deterministic worker-name -> worker lookup
  orchestrator.py  run() routes, plan() plans, execute_plan() runs a plan
  llm/
    base.py        Message, Usage, LLMRequest, LLMResponse, LLMClient (ABC)
    deepseek.py    DeepSeek adapter over the openai SDK
    factory.py     provider registry -> build_llm_client()
  planner/
    base.py        Planner contract + planner errors
    llm.py         LLMPlanner: prompt -> JSON -> validated Plan
  workers/
    base.py        BaseWorker contract + WorkerError
    registry.py    WorkerRegistry, maps worker names to worker instances
    research.py    ResearchWorker, a single LLM call with no tools
    coding.py      CodingWorker, coding-oriented prompt, no code execution
tests/             unit tests (network-free)
```
## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env   # then fill in DEEPSEEK_API_KEY
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `deepseek` | Which adapter the registry builds |
| `DEEPSEEK_API_KEY` | - | Credential (required at call time) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible endpoint |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Default model |
| `LLM_TEMPERATURE` | `0.0` | Default sampling temperature |
| `LLM_MAX_TOKENS` | `2048` | Default completion budget |
| `LLM_TIMEOUT_S` | `60` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | SDK-level retry count |

## Usage

```python
from orchestrator_worker.config import Settings
from orchestrator_worker.llm import LLMRequest, Message, build_llm_client

settings = Settings.from_env()
client = build_llm_client(settings)

response = client.complete(
    LLMRequest(messages=[Message.user("Say hello in one word.")])
)
print(response.content, response.usage.total_tokens)
```

### Phase 2 chain

```python
from orchestrator_worker.config import Settings
from orchestrator_worker.llm import build_llm_client
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.workers import CodingWorker, ResearchWorker, WorkerRegistry

client = build_llm_client(Settings.from_env())

registry = WorkerRegistry()
registry.register(ResearchWorker(client))
registry.register(CodingWorker(client))

orchestrator = Orchestrator(registry)

print(orchestrator.run("What is DeepSeek V3?").worker_name)              # 'research'
print(orchestrator.run("Sort dicts by key.", worker_name="coding").worker_name)  # 'coding'
```

`Orchestrator(single_worker)` also still works: it wraps that one worker in a
registry and makes it the default target.

### Phase 3 planning

```python
from orchestrator_worker.config import Settings
from orchestrator_worker.llm import build_llm_client
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.planner import LLMPlanner
from orchestrator_worker.workers import CodingWorker, ResearchWorker, WorkerRegistry

client = build_llm_client(Settings.from_env())

registry = WorkerRegistry()
registry.register(ResearchWorker(client))
registry.register(CodingWorker(client))

planner = LLMPlanner(client, available_workers=registry.names())
orchestrator = Orchestrator(registry, planner=planner)

plan = orchestrator.plan("research asyncio and write a small example")
print(plan.goal)
for step in plan.steps:
    print(step.id, step.worker_name, step.task)
```

The planner is given the worker names that exist (`registry.names()`) and
rejects any plan that references a different worker. It receives only that
read-only list of names, so it cannot look up or run a worker.

### Phase 4B/4C plan execution

```python
results = orchestrator.execute_plan(plan)
for result in results:
    print(result.step_id, result.worker_name, result.state.final_result)

context = orchestrator.execute_plan_context(plan)
print(context.plan.goal, len(context.results))
```

`execute_plan()` returns one `ExecutionResult` per step, in the same order as
`plan.steps`. Each step is resolved by `step.worker_name` through the router
and executed with a single `worker.execute(step.task)` call, so steps are
never merged and never run in parallel. An `ExecutionResult` keeps the
`step_id`, `task` and `worker_name` of its step next to the `AgentState` the
worker produced, and `execute_plan_context()` returns the same results as an
`ExecutionContext` next to the plan they came from. A step that names an
unknown worker raises `RegistryError`, and a failing worker's `WorkerError`
propagates unchanged.

### Phase 4D aggregation

```python
from orchestrator_worker.aggregators import LLMAggregator

aggregator = LLMAggregator(client)
orchestrator = Orchestrator(registry, planner=planner, aggregator=aggregator)

context = orchestrator.execute_plan_context(orchestrator.plan("research asyncio"))
print(orchestrator.aggregate(context))
```

The aggregator receives the whole `ExecutionContext` and writes the final
answer from results that already exist. It never plans, routes, executes or
retries anything, and it lays the steps out in `plan` order regardless of how
`context.results` happens to be ordered. It reaches the model only through
`LLMClient`, so the aggregation model can be swapped without touching the
aggregator logic. Retry, parallel execution and tool calling are still out of
scope.

### Phase 4E evaluation and retry

```python
from orchestrator_worker.evaluators import LLMEvaluator

evaluator = LLMEvaluator(client)
verdict = evaluator.evaluate(context)

print(verdict.passed, verdict.reason)
```

The evaluator inspects a finished `ExecutionContext` and returns an
`EvaluationResult`: whether those results are enough to produce the final
answer, plus a reason. It lays the steps out in `plan` order, reaches the model
only through `LLMClient`, and never plans, executes, retries or modifies the
context. It is a separate component from the aggregator.

#### Evaluator-driven retry

```python
orchestrator = Orchestrator(registry, evaluator=LLMEvaluator(client))

context = orchestrator.execute_plan_with_retry(plan, max_retries=1)
```

`max_retries` counts the extra attempts after the first execution, so
`max_retries=1` runs the plan at most twice. Every attempt re-executes the
whole plan from scratch and re-evaluates it; an individual step is never
retried on its own, because an `EvaluationResult` carries no failing step id.
Only `passed=False` starts another attempt, and once the budget is spent the
method raises `RetryError` carrying the last reason. Worker, registry and
evaluator errors propagate unchanged instead of turning into a retry, the plan
is never re-planned, and the aggregator is not called automatically.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```