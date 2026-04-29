from domains.building.adapter import BuildingAdapter
from domains.building.safety import simulate_building_step
from core.pipeline import run_abci
from core.types import ValidationResult, DecisionResult


def make_adapter() -> BuildingAdapter:
    return BuildingAdapter()


def make_trusted_state(**overrides):
    state = {
        "role": "security_operator",
        "badge_valid": True,
        "fire_alarm": False,
        "building_alarm": False,
        "lockdown_active": False,
        "after_hours": False,
        "authorized_zones": ["lobby", "lab_a", "server_room"],
        "restricted_zones": ["server_room", "mechanical_room"],
    }
    state.update(overrides)
    return state


def make_intent(**overrides):
    intent = {
        "action": "grant_access",
        "target": "lab_a",
        "params": {
            "zone_id": "lab_a",
            "duration_minutes": 15,
            "access_level": "employee",
        },
        "justification": "Badge reader request",
        "evidence": ["badge valid"],
    }
    intent.update(overrides)
    return intent


def test_validate_accepts_valid_grant_access_intent():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent()

    result = adapter.validate(state, intent)

    assert isinstance(result, ValidationResult)
    assert result.accepted is True
    assert result.validated_intent["action"] == "grant_access"
    assert result.validated_intent["params"]["zone_id"] == "lab_a"


def test_validate_rejects_invalid_badge_for_grant_access():
    adapter = make_adapter()
    state = make_trusted_state(badge_valid=False)
    intent = make_intent()

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "invalid_badge"


def test_validate_rejects_unauthorized_zone():
    adapter = make_adapter()
    state = make_trusted_state(authorized_zones=["lobby"])
    intent = make_intent(target="server_room", params={"zone_id": "server_room"})

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "restricted_zone_unauthorized"


def test_validate_rejects_lockdown_from_unprivileged_role():
    adapter = make_adapter()
    state = make_trusted_state(role="visitor")
    intent = make_intent(action="lockdown", params={})

    result = adapter.validate(state, intent)

    assert result.accepted is False
    assert result.reason == "insufficient_role_for_lockdown"


def test_decide_forces_evacuation_during_fire_alarm():
    adapter = make_adapter()
    state = make_trusted_state(fire_alarm=True)
    validated_intent = {
        "action": "deny",
        "target": "lobby",
        "params": {"zone_id": "lobby", "duration_minutes": None, "access_level": None},
        "justification": None,
        "evidence": None,
    }

    result = adapter.decide(state, validated_intent)

    assert isinstance(result, DecisionResult)
    assert result.action == "evacuate"
    assert result.audit["decision_reason"] == "fire_alarm_evacuation"


def test_decide_requires_confirmation_for_after_hours_access():
    adapter = make_adapter()
    state = make_trusted_state(role="employee", after_hours=True)
    validated_intent = make_intent()

    result = adapter.decide(state, validated_intent)

    assert result.action == "request_confirmation"
    assert result.audit["decision_reason"] == "after_hours_confirmation"


def test_run_abci_uses_deny_fallback_on_validation_failure():
    adapter = make_adapter()
    state = make_trusted_state()
    intent = make_intent(action="open_everything")

    result = run_abci(adapter, state, intent)

    assert result.action == "deny"
    assert result.fallback_used is True
    assert result.audit["reject_reason"] == "invalid_action"


def test_safety_flags_grant_access_during_lockdown():
    state = make_trusted_state(lockdown_active=True)
    intent = make_intent()

    step = simulate_building_step(state, "grant_access", intent)

    assert step["unsafe"] is True
    assert "grant_access_during_lockdown" in step["hazard_reasons"]
