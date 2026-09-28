# Orchestrator-Worker

A multi-agent backend prototype. Target request flow:

```
User -> Orchestrator -> Planner -> Worker Router -> Workers
     -> Evaluator -> Aggregator -> Final Answer
```

Current status: **Phase 4A**. Three capabilities are implemented and tested.

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
Plan -> Orchestrator -> Router -> Worker (per step) -> AgentState
```

`plan()` only produces a `Plan`: the planner never executes a step and has no
access to the worker registry. `execute_plan()` then walks `plan.steps` in
order, resolving each `step.worker_name` through a deterministic router (no
LLM involved) and calling that worker once per step. Execution is strictly
sequential: there is no retry, no evaluator, no aggregator and no parallel
execution.

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

### Phase 4A plan execution

```python
states = orchestrator.execute_plan(plan)
for state in states:
    print(state.worker_name, state.final_result)
```

`execute_plan()` returns one `AgentState` per step, in the same order as
`plan.steps`. Each step is resolved by `step.worker_name` through the router
and executed with a single `worker.execute(step.task)` call, so steps are
never merged and never run in parallel. A step that names an unknown worker
raises `RegistryError`, and a failing worker's `WorkerError` propagates
unchanged.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```