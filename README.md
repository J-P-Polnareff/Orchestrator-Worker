# Orchestrator-Worker

A multi-agent backend prototype. Target request flow:

```
User -> Orchestrator -> Planner -> Worker Router -> Workers
     -> Evaluator -> Aggregator -> Final Answer
```

Current status: **Phase 1**. Packaging, configuration, the provider-agnostic
LLM layer, and a minimal end-to-end chain work and are tested:

```
User Task -> Orchestrator -> ResearchWorker -> Worker Result -> Final Result
```

Routing is fixed to a single injected worker. Planner, Worker Router,
Evaluator and Aggregator are not implemented yet.

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
  orchestrator.py  Phase 1 orchestrator with fixed routing
  llm/
    base.py        Message, Usage, LLMRequest, LLMResponse, LLMClient (ABC)
    deepseek.py    DeepSeek adapter over the openai SDK
    factory.py     provider registry -> build_llm_client()
  workers/
    base.py        BaseWorker contract + WorkerError
    research.py    ResearchWorker, a single LLM call with no tools
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

### Phase 1 chain

```python
from orchestrator_worker.config import Settings
from orchestrator_worker.llm import build_llm_client
from orchestrator_worker.orchestrator import Orchestrator
from orchestrator_worker.workers import ResearchWorker

client = build_llm_client(Settings.from_env())
state = Orchestrator(ResearchWorker(client)).run("What is DeepSeek V3?")

print(state.worker_name)    # 'research'
print(state.final_result)
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```