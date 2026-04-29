from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: Optional[str] = None
    validated_intent: Optional[dict[str, Any]] = None

    @staticmethod
    def accept(validated_intent: dict[str, Any]) -> "ValidationResult":
        return ValidationResult(
            accepted=True,
            reason=None,
            validated_intent=validated_intent,
        )

    @staticmethod
    def reject(reason: str) -> "ValidationResult":
        return ValidationResult(
            accepted=False,
            reason=reason,
            validated_intent=None,
        )


@dataclass(frozen=True)
class DecisionResult:
    action: str
    fallback_used: bool
    audit: dict[str, Any] = field(default_factory=dict)