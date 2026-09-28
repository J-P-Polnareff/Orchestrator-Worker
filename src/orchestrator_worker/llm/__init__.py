"""Provider-agnostic LLM layer.

Public surface for agent code: import from here, never from a concrete adapter.
"""

from .base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    Usage,
)
from .factory import (
    available_providers,
    build_llm_client,
    register_provider,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "Role",
    "Usage",
    "available_providers",
    "build_llm_client",
    "register_provider",
]