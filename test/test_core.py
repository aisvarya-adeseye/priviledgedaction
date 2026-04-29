from core.domain_adapter import DomainAdapter
from core.pipeline import run_abci
from core.types import ValidationResult, DecisionResult


class DummyAdapter(DomainAdapter):
    name = "dummy"

    def validate(self, trusted_state, intent):
        if intent.get("verb") == "bad":
            return ValidationResult.reject("bad_verb")
        return ValidationResult.accept(intent)

    def decide(self, trusted_state, validated_intent):
        return DecisionResult(
            action=validated_intent["verb"],
            fallback_used=False,
            audit={"stage": "decide"},
        )

    def safe_fallback(self, trusted_state):
        return "halt"


def test_accept_path():
    adapter = DummyAdapter()
    result = run_abci(adapter, {"mode": "normal"}, {"verb": "continue"})
    assert result.action == "continue"
    assert result.fallback_used is False
    assert result.audit["accepted"] is True


def test_reject_path():
    adapter = DummyAdapter()
    result = run_abci(adapter, {"mode": "normal"}, {"verb": "bad"})
    assert result.action == "halt"
    assert result.fallback_used is True
    assert result.audit["accepted"] is False
    assert result.audit["reject_reason"] == "bad_verb"