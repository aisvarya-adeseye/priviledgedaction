from domains.valve.adapter import ValveAdapter
from domains.valve.safety import simulate_valve_step
from core.pipeline import run_abci
from core.types import ValidationResult, DecisionResult


def make_adapter() -> ValveAdapter:
    return ValveAdapter()


def make_trusted_state(**overrides):
    state = {
        "role": "process_engineer",
        "authorized_valves": ["valve_1", "valve_2", "valve_3"],
        "valve_locked": False,
        "maintenance_mode": False,
        "manual_mode": False,
        "emergency_shutdown": False,
        "leak_detected": False,
        "downstream_blocked": False,
        "vent_path_clear": True,
        "pressure_psi": 40.0,
        "max_safe_pressure_psi": 100.0,
    }
    state.update(overrides)
    return state


def make_intent(**overrides):
    intent = {
        "action": "open",
        "target": "valve_1",
        "params": {
            "valve_id": "valve_1",
            "flow_rate": 12.5,
            "duration_seconds": 30,
        },
        "justification": "Process startup request",
        "evidence": ["pressure normal"],
    }
    intent.update(overrides)
    return intent


def test_validate_accepts_valid_open_intent():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent()

    result = adapter.validate(state, intent)

    assert isinstance(result, ValidationResult)
    assert result.accepted is True
    assert result.validated_intent["action"] == "open"
    assert result.validated_intent["params"]["valve_id"] == "valve_1"


def test_validate_rejects_unauthorized_valve():
    adapter = make_adapter()
    state = make_trusted_state(authorized_valves=["valve_1"])
    intent = make_intent(target="valve_9", params={"valve_id": "valve_9"})

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "unauthorized_valve"


def test_validate_rejects_open_when_valve_locked():
    adapter = make_adapter()
    state = make_trusted_state(valve_locked=True)
    intent = make_intent()

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "valve_locked"


def test_validate_rejects_negative_flow_rate():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(params={"valve_id": "valve_1", "flow_rate": -1})

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "flow_rate_negative"


def test_decide_forces_close_on_emergency_shutdown():
    adapter = make_adapter()
    state = make_trusted_state(emergency_shutdown=True)
    validated_intent = {
        "action": "hold",
        "target": "valve_1",
        "params": {"valve_id": "valve_1", "flow_rate": None, "duration_seconds": None},
        "justification": None,
        "evidence": None,
    }

    result = adapter.decide(state, validated_intent)

    assert isinstance(result, DecisionResult)
    assert result.action == "close"
    assert result.audit["decision_reason"] == "emergency_shutdown_close"


def test_decide_remaps_open_to_vent_when_pressure_high():
    adapter = make_adapter()
    state = make_trusted_state(pressure_psi=125.0, max_safe_pressure_psi=100.0)
    validated_intent = make_intent()

    result = adapter.decide(state, validated_intent)

    assert result.action == "vent"
    assert result.audit["decision_reason"] == "high_pressure_remap"


def test_decide_requires_confirmation_for_manual_mode_open():
    adapter = make_adapter()
    state = make_trusted_state(manual_mode=True)
    validated_intent = make_intent()

    result = adapter.decide(state, validated_intent)

    assert result.action == "request_confirmation"
    assert result.audit["decision_reason"] == "manual_mode_confirmation"


def test_run_abci_uses_hold_fallback_on_validation_failure():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(action="launch")

    result = run_abci(adapter, state, intent)

    assert result.action == "hold"
    assert result.fallback_used is True
    assert result.audit["reject_reason"] == "invalid_action"


def test_safety_flags_open_above_max_safe_pressure():
    state = make_trusted_state(pressure_psi=125.0, max_safe_pressure_psi=100.0)
    intent = make_intent()

    step = simulate_valve_step(state, "open", intent)

    assert step["unsafe"] is True
    assert "open_above_max_safe_pressure" in step["hazard_reasons"]
