from abc import ABC, abstractmethod
from typing import Any

from core.types import ValidationResult, DecisionResult


class DomainAdapter(ABC):
    """
    Abstract interface that every case-study domain must implement.

    The whole point of this class is to keep the ABCI pipeline generic:
    the core pipeline knows nothing about robots, valves, or buildings.
    It only knows how to call this interface.
    """

    # Human-readable domain name, e.g. "robot", "valve", "building".
    name: str

    @abstractmethod
    def validate(
        self,
        trusted_state: dict[str, Any],
        intent: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate and normalize a tentative intent.

        Inputs:
            trusted_state:
                Trusted control-plane state for the domain.
            intent:
                Parsed intent produced upstream (eventually by an LLM,
                but for now possibly hand-written in tests).

        Output:
            ValidationResult.accept(...) if the intent is valid and safe
            to pass to decision logic, otherwise ValidationResult.reject(...).
        """
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        trusted_state: dict[str, Any],
        validated_intent: dict[str, Any],
    ) -> DecisionResult:
        """
        Compute the final privileged action from trusted state and a
        validated intent only.

        Important:
            This method should never depend on raw untrusted text.
            It should only consume trusted_state + validated_intent.
        """
        raise NotImplementedError

    @abstractmethod
    def safe_fallback(self, trusted_state: dict[str, Any]) -> str:
        """
        Return the domain's conservative safe action when validation fails.

        Examples:
            robot    -> "halt"
            valve    -> "hold"
            building -> "deny"
        """
        raise NotImplementedError