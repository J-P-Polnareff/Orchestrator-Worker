# Orchestrator-Worker

A multi-agent backend prototype. Target request flow:

```
User -> Orchestrator -> Planner -> Worker Router -> Workers
     -> Evaluator -> Aggregator -> Final Answer
```

Current status: **Phase 5C-4A (tool-enabled worker boundary)**. On top of the
tool execution loop, a `ToolEnabledWorker` now holds its own allowed `Tool`
definitions and runs a task through an injected `ToolLoop`, turning the final
`LLMResponse` back into a worker result. The loop stays the only tool runtime
entry point, and the flow below is unchanged: no existing worker, the
orchestrator or the application is wired to tools yet.

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

Pipeline - `Orchestrator.run_pipeline(task, max_retries=...)`:

```
User Task -> Planner -> Plan -> execute_plan_with_retry -> ExecutionContext
          -> Aggregator -> Final Answer
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
`passed=False`. `run_pipeline()` chains these layers explicitly - plan once,
execute with retry, then aggregate only the context that passed - while `run()`
keeps its original single-worker behaviour. There is no replanning, no targeted
per-step retry, no automatic aggregation and no parallel execution.

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
  orchestrator.py  run() routes, plan() plans, execute_plan() runs a plan,
                   run_pipeline() chains the full plan/execute/aggregate flow
  application.py   build_orchestrator(): concrete wiring for a ready Orchestrator
  tools.py         Tool / ToolCall / ToolResult + ToolRegistry and ToolExecutor
  tool_loop.py     ToolLoop: LLM <-> ToolExecutor until the model stops
  llm/
    base.py        Message (tool calls/results), LLMRequest (tools),
                   LLMResponse (tool_calls), Usage, LLMClient (ABC)
    deepseek.py    DeepSeek adapter: Tool definitions out, parsed ToolCalls back
    factory.py     provider registry -> build_llm_client()
  planner/
    base.py        Planner contract + planner errors
    llm.py         LLMPlanner: prompt -> JSON -> validated Plan
  workers/
    base.py        BaseWorker contract + WorkerError
    registry.py    WorkerRegistry, maps worker names to worker instances
    research.py    ResearchWorker, a single LLM call with no tools
    coding.py      CodingWorker, coding-oriented prompt, no code execution
    tool_enabled.py  ToolEnabledWorker: Worker <-> ToolLoop boundary
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

### Phase 4F pipeline

```python
from orchestrator_worker.aggregators import LLMAggregator
from orchestrator_worker.evaluators import LLMEvaluator
from orchestrator_worker.planner import LLMPlanner

orchestrator = Orchestrator(
    registry,
    planner=LLMPlanner(client, available_workers=registry.names()),
    evaluator=LLMEvaluator(client),
    aggregator=LLMAggregator(client),
)

answer = orchestrator.run_pipeline("research asyncio and write a small example")
print(answer)
```

`run_pipeline()` is the explicit end-to-end flow: it calls `plan()` once, hands
that same `Plan` to `execute_plan_with_retry()`, and aggregates only the
`ExecutionContext` that passed evaluation. The planner is never re-invoked, no
attempt is aggregated more than once, and errors (`PlannerError`,
`WorkerError`, `RegistryError`, `EvaluatorError`, `RetryError`,
`AggregatorError`, `OrchestratorError`) propagate unchanged. `run()` still works
exactly as before.

### Phase 5A application wiring

```python
from orchestrator_worker.application import build_orchestrator

orchestrator = build_orchestrator()        # builds the configured provider

# or inject a client explicitly (tests, alternate providers):
orchestrator = build_orchestrator(client)

answer = orchestrator.run_pipeline("research asyncio and write a small example")
```

`build_orchestrator()` is the composition root. It creates the LLM client (or
accepts an injected one), registers `ResearchWorker` and `CodingWorker` in a
`WorkerRegistry`, and builds the `LLMPlanner` (given the registry's worker
names), `LLMEvaluator` and `LLMAggregator` - all sharing that same `LLMClient` -
then injects them into an `Orchestrator`. It only constructs and injects: it
never plans, executes or calls a model, so building is side-effect free.
`orchestrator.py` still imports no concrete provider or worker; only this module
depends on concrete implementations. Phase 5A does not add provider routing, a
Claude adapter, real-API integration tests or tool calling.

### Phase 5C-1 tool foundation

```python
from orchestrator_worker.tools import Tool, ToolCall, ToolExecutor, ToolRegistry

def add(args):            # handler: any callable over the parsed arguments
    return args["a"] + args["b"]

calculator = Tool(
    name="calculator",
    description="Perform a simple arithmetic operation.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
    handler=add,
)

registry = ToolRegistry()
registry.register(calculator)

executor = ToolExecutor(registry)
result = executor.execute(
    ToolCall(id="call-1", name="calculator", arguments={"a": 1, "b": 2})
)
print(result.output, result.is_error)     # "3 False"
```

`Tool`, `ToolCall` and `ToolResult` are frozen, validated, provider-neutral data
objects; `parameters` is stored as a plain mapping and is not schema-validated
yet. `ToolRegistry` rejects duplicate names, never falls back on an unknown
lookup, and returns sorted names plus a read-only tuple of tools. `ToolExecutor`
resolves the tool by name, calls `handler(arguments)` and normalizes the result
with `str()`; an unknown tool raises `ToolRegistryError` and a handler failure
raises `ToolExecutionError` with the original exception as `__cause__`.

This layer is deliberately not connected to the model yet: there is no LLM tool
calling, no tool loop, and no change to `LLMClient`, the workers, the planner,
the evaluator, the aggregator or the orchestrator. Offering tools to a provider
and driving the call/execute cycle from a model response is the next step.

### Phase 5C-2 LLM tool-call contract

```python
from orchestrator_worker.llm import LLMRequest, Message
from orchestrator_worker.tools import Tool, ToolCall

request = LLMRequest(
    messages=[Message.user("add 1 and 2")],
    tools=(calculator,),                      # Tool definitions
)
response = client.complete(request)

for call in response.tool_calls:              # provider-neutral ToolCall
    print(call.id, call.name, call.arguments)

# continue the conversation with the call and its result
messages = [
    Message.user("add 1 and 2"),
    Message(role="assistant", content="", tool_calls=response.tool_calls),
    Message.tool("3", response.tool_calls[0].id),
]
```

`LLMRequest.tools` and `LLMResponse.tool_calls` default to empty tuples, so every
existing construction keeps working. `Message` gained `tool_calls` (allowed only
on an assistant message) and `tool_call_id` (allowed only on a tool message,
which must set it); plain system/user/assistant messages serialize exactly as
before. There is no `tool_choice` yet.

The DeepSeek adapter owns the translation: it sends each `Tool` as an
OpenAI-compatible `{"type": "function", "function": {...}}` entry (never the
handler) and parses `message.tool_calls` back into `ToolCall` objects, requiring
`arguments` to be a JSON *object*. Missing or empty ids and names, unsupported
tool types, malformed JSON and non-object JSON all raise `LLMError` instead of
being repaired, and no provider SDK object leaves the adapter.

The execution loop itself now lives in `tool_loop.py` (Phase 5C-3); still out of
scope here are worker integration, orchestrator integration and MCP.

### Phase 5C-3 Tool execution loop

```python
from orchestrator_worker.llm import LLMRequest, Message
from orchestrator_worker.tool_loop import ToolLoop
from orchestrator_worker.tools import ToolExecutor

loop = ToolLoop(llm_client, ToolExecutor(registry), max_rounds=8)
response = loop.run(
    LLMRequest(messages=[Message.user("add 1 and 2")], tools=(calculator,))
)
response.content   # final answer, once the model stops calling tools
```

`ToolLoop` is a standalone runtime component, not part of a worker or the
orchestrator. One round is one `LLMClient.complete()` call: a response without
tool calls is returned as-is, otherwise the loop appends an assistant tool-call
message plus one tool message per result and calls the model again with the same
`tools` and a longer history. Several tool calls in one response run strictly
sequentially, in response order - there is no async or parallel execution.

`max_rounds` (default 8) bounds the number of model calls; a tool call that
arrives in the last permitted round raises `ToolLoopError` before its result
could ever reach the model. A failing tool (`ToolExecutionError`) becomes an
`is_error=True` tool message so the model can react, while an unknown tool
(`ToolRegistryError`) and an `LLMError` propagate unchanged: the loop never
retries the model, never replans and never triggers the Orchestrator retry.

Still out of scope there: no orchestrator integration, no `tool_choice` and no
MCP.

### Phase 5C-4A Tool-enabled worker boundary

```python
from orchestrator_worker.workers.tool_enabled import ToolEnabledWorker

class ResearchToolWorker(ToolEnabledWorker):
    name = "research"
    system_prompt = "You are a research worker that may use tools."

worker = ResearchToolWorker(tool_loop, tools=(calculator,))   # ToolLoop injected
answer = worker.execute("add 1 and 2 and explain the result")
```

`ToolEnabledWorker` is the Worker <-> ToolLoop seam. It stores the exact `Tool`
tuple it is allowed to request, builds an `LLMRequest` with those tools plus its
system/user messages, and calls `tool_loop.run(...)` once. The loop owns every
model call and every tool result, so the worker never executes a tool, never
looks one up and never re-runs a request.

The `ToolLoop` and the allowed tools are both injected: the worker builds no
loop, no registry and no model client. Subclasses only supply their identity and
prompt (`name`, `system_prompt`, or a full `build_request` override); the base
class handles the tool wiring and the `LLMResponse` -> `str` conversion.
`ResearchWorker` and `CodingWorker` are untouched and stay tool-free.

Still out of scope: no concrete research/coding tool worker, no orchestrator
wiring and no application wiring - that is Phase 5C-4B.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```