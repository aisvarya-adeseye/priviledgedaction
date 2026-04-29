from typing import Any

from core.domain_adapter import DomainAdapter
from core.parser_interface import IntentParser
from core.types import DecisionResult


def run_abci(
    domain_adapter: DomainAdapter,
    trusted_state: dict[str, Any],
    intent: dict[str, Any],
) -> DecisionResult:
    """
    Run the shared ABCI pipeline from an already-parsed intent.
    """
    validation = domain_adapter.validate(trusted_state, intent)

    if not validation.accepted:
        return DecisionResult(
            action=domain_adapter.safe_fallback(trusted_state),
            fallback_used=True,
            audit={
                "domain": domain_adapter.name,
                "accepted": False,
                "reject_reason": validation.reason,
                "validated_intent": None,
                "tentative_intent": intent,
            },
        )

    decision = domain_adapter.decide(
        trusted_state,
        validation.validated_intent,
    )

    merged_audit = {
        **decision.audit,
        "domain": domain_adapter.name,
        "accepted": True,
        "reject_reason": None,
        "validated_intent": validation.validated_intent,
        "tentative_intent": intent,
    }

    return DecisionResult(
        action=decision.action,
        fallback_used=decision.fallback_used,
        audit=merged_audit,
    )


def run_abci_from_text(
    domain_adapter: DomainAdapter,
    parser: IntentParser,
    trusted_state: dict[str, Any],
    raw_text: str,
    target: str | None = None,
    evidence: list[str] | None = None,
) -> DecisionResult:
    """
    Run the shared ABCI pipeline directly from raw text.
    """
    try:
        tentative_intent, raw_parser_output = parser.parse(
            trusted_state=trusted_state,
            raw_text=raw_text,
            target=target,
            evidence=evidence,
        )
    except Exception as exc:
        return DecisionResult(
            action=domain_adapter.safe_fallback(trusted_state),
            fallback_used=True,
            audit={
                "domain": domain_adapter.name,
                "accepted": False,
                "reject_reason": "parser_error",
                "validated_intent": None,
                "tentative_intent": None,
                "raw_input": raw_text,
                "parser_raw_output": None,
                "parser_exception": str(exc),
            },
        )

    decision = run_abci(
        domain_adapter=domain_adapter,
        trusted_state=trusted_state,
        intent=tentative_intent,
    )

    merged_audit = {
        **decision.audit,
        "raw_input": raw_text,
        "parser_raw_output": raw_parser_output,
    }

    return DecisionResult(
        action=decision.action,
        fallback_used=decision.fallback_used,
        audit=merged_audit,
    )
