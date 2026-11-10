"""Abstract interface for model providers (Groq and Stand-in)."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from relay.schemas import ProviderResponse


class ModelProvider(ABC):
    @property
    def provider_name(self) -> str:
        """Identifying name of the provider ('standin' or 'groq')."""
        return "generic"

    @property
    def model_name(self) -> str:
        """Underlying model identifier exposed by this provider."""
        return "generic-model"

    @property
    def is_simulated(self) -> bool:
        """True if responses and telemetry are simulated; False for real APIs."""
        return True

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> ProviderResponse:
        """Executes a non-streaming chat completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs,
    ) -> AsyncGenerator[Tuple[str, Optional[ProviderResponse]], None]:
        """Executes a streaming chat completion.
        Yields (chunk_delta, None) during generation, and ('', final_provider_response) at termination.
        """
        pass
