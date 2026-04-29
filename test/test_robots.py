from domains.robot.adapter import RobotAdapter
from core.pipeline import run_abci
from core.types import ValidationResult, DecisionResult


def make_adapter() -> RobotAdapter:
    return RobotAdapter()


def make_trusted_state(**overrides):
    """
    Helper to build a default trusted robot state for tests.
    Individual tests can override only the fields they care about.
    """
    state = {
        "role": "operator",
        "control_mode": "normal",
        "estop": False,
        "path_blocked": False,
        "allowed_routes": ["r1", "r2", "r3"],
    }
    state.update(overrides)
    return state


def make_intent(**overrides):
    """
    Helper to build a default tentative intent for tests.
    Individual tests can override fields as needed.
    """
    intent = {
        "verb": "continue",
        "target": "robot_1",
        "params": {},
        "justification": "Proceed with task",
        "evidence": ["operator request"],
    }
    intent.update(overrides)
    return intent


# ---------------------------------------------------------------------------
# Validation-only tests
# ---------------------------------------------------------------------------

def test_validate_accepts_valid_continue_intent():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent()

    result = adapter.validate(state, intent)

    assert isinstance(result, ValidationResult)
    assert result.accepted is True
    assert result.reason is None
    assert result.validated_intent["verb"] == "continue"
    assert result.validated_intent["target"] == "robot_1"


def test_validate_rejects_non_dict_intent():
    adapter = make_adapter()
    state = make_trusted_state()

    result = adapter.validate(state, "not a dict")

    assert result.accepted is False
    assert result.reason == "intent_not_dict"


def test_validate_rejects_missing_verb():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent()
    intent.pop("verb")

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "missing_verb"


def test_validate_rejects_non_string_verb():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(verb=123)

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "verb_not_string"


def test_validate_rejects_invalid_verb():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(verb="open_door")

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "invalid_verb"


def test_validate_normalizes_verb_and_target():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(verb="  CONTINUE  ", target="  ROBOT_1  ")

    result = adapter.validate(state, intent)

    assert result.accepted is True
    assert result.validated_intent["verb"] == "continue"
    assert result.validated_intent["target"] == "robot_1"


def test_validate_rejects_non_string_target():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(target=42)

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "target_not_string"


def test_validate_rejects_non_dict_params():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(params="bad")

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "params_not_dict"


def test_validate_rejects_non_string_justification():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(justification=999)

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "justification_not_string"


def test_validate_rejects_non_list_evidence():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(evidence="not-a-list")

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "evidence_not_list"


def test_validate_rejects_non_string_route_id():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(
        verb="reroute",
        params={"route_id": 123}
    )

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "route_id_not_string"


def test_validate_rejects_non_string_speed_band():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(
        params={"speed_band": 2}
    )

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "speed_band_not_string"


def test_validate_rejects_reroute_without_route_id():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(verb="reroute", params={})

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "reroute_missing_route_id"


def test_validate_rejects_unauthorized_route():
    adapter = make_adapter()
    state = make_trusted_state(allowed_routes=["r1", "r2"])
    intent = make_intent(
        verb="reroute",
        params={"route_id": "r99"}
    )

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "unauthorized_route"


def test_validate_rejects_continue_when_estop_active():
    adapter = make_adapter()
    state = make_trusted_state(estop=True)
    intent = make_intent(verb="continue")

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "estop_blocks_motion"


def test_validate_rejects_reroute_when_estop_active():
    adapter = make_adapter()
    state = make_trusted_state(estop=True)
    intent = make_intent(
        verb="reroute",
        params={"route_id": "r1"}
    )

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "estop_blocks_motion"


def test_validate_allows_halt_when_estop_active():
    adapter = make_adapter()
    state = make_trusted_state(estop=True)
    intent = make_intent(verb="halt")

    result = adapter.validate(state, intent)

    assert result.accepted is True
    assert result.validated_intent["verb"] == "halt"


def test_validate_allows_request_confirmation_when_estop_active():
    adapter = make_adapter()
    state = make_trusted_state(estop=True)
    intent = make_intent(verb="request_confirmation")

    result = adapter.validate(state, intent)

    assert result.accepted is True
    assert result.validated_intent["verb"] == "request_confirmation"


# ---------------------------------------------------------------------------
# Decision-only tests
# ---------------------------------------------------------------------------

def test_decide_passthrough_for_valid_continue():
    adapter = make_adapter()
    state = make_trusted_state()
    validated_intent = {
        "verb": "continue",
        "target": "robot_1",
        "params": {"route_id": None, "speed_band": None},
        "justification": "Proceed",
        "evidence": ["operator request"],
    }

    result = adapter.decide(state, validated_intent)

    assert isinstance(result, DecisionResult)
    assert result.action == "continue"
    assert result.fallback_used is False
    assert result.audit["decision_reason"] == "validated_passthrough"


def test_decide_forces_halt_when_estop_active():
    adapter = make_adapter()
    state = make_trusted_state(estop=True)
    validated_intent = {
        "verb": "continue",
        "target": "robot_1",
        "params": {"route_id": None, "speed_band": None},
        "justification": None,
        "evidence": None,
    }

    result = adapter.decide(state, validated_intent)

    assert result.action == "halt"
    assert result.fallback_used is False
    assert result.audit["decision_reason"] == "estop_active"


def test_decide_remaps_continue_to_reroute_when_path_blocked():
    adapter = make_adapter()
    state = make_trusted_state(path_blocked=True)
    validated_intent = {
        "verb": "continue",
        "target": "robot_1",
        "params": {"route_id": None, "speed_band": None},
        "justification": None,
        "evidence": None,
    }

    result = adapter.decide(state, validated_intent)

    assert result.action == "reroute"
    assert result.fallback_used is False
    assert result.audit["decision_reason"] == "blocked_path_remap"


def test_decide_requires_confirmation_for_reroute_in_manual_mode():
    adapter = make_adapter()
    state = make_trusted_state(control_mode="manual")
    validated_intent = {
        "verb": "reroute",
        "target": "robot_1",
        "params": {"route_id": "r1", "speed_band": None},
        "justification": None,
        "evidence": None,
    }

    result = adapter.decide(state, validated_intent)

    assert result.action == "request_confirmation"
    assert result.fallback_used is False
    assert result.audit["decision_reason"] == "manual_mode_confirmation"


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------

def test_run_abci_returns_fallback_on_validation_failure():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(verb="invalid")

    result = run_abci(adapter, state, intent)

    assert result.action == "halt"
    assert result.fallback_used is True
    assert result.audit["accepted"] is False
    assert result.audit["reject_reason"] == "invalid_verb"
    assert result.audit["validated_intent"] is None


def test_run_abci_returns_decision_on_valid_input():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(verb="continue")

    result = run_abci(adapter, state, intent)

    assert result.action == "continue"
    assert result.fallback_used is False
    assert result.audit["accepted"] is True
    assert result.audit["reject_reason"] is None
    assert result.audit["validated_intent"]["verb"] == "continue"


def test_run_abci_remaps_continue_to_reroute_when_blocked():
    adapter = make_adapter()
    state = make_trusted_state(path_blocked=True)
    intent = make_intent(verb="continue")

    result = run_abci(adapter, state, intent)

    assert result.action == "reroute"
    assert result.fallback_used is False
    assert result.audit["accepted"] is True
    assert result.audit["decision_reason"] == "blocked_path_remap"