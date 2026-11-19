"""Conservative event planning: consume the existing planner, never guess order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EventPlan:
    actions: tuple[str, ...] = ()
    reason: str = "no_explicit_sequence"

    @property
    def active(self):
        return bool(self.actions)


def build_event_plan(query, max_events=6):
    events = query.get("events") or []
    constraints = query.get("temporal_constraints") or []
    if not constraints:
        return EventPlan()
    if not isinstance(events, list) or not isinstance(constraints, list):
        return EventPlan(reason="invalid_sequence")
    actions = [event["action"].strip() for event in events
               if isinstance(event, dict) and isinstance(event.get("action"), str) and event["action"].strip()]
    # Multiple/partial orders need a graph planner; do not silently flatten them.
    if len(constraints) != 1 or not isinstance(constraints[0], dict) or constraints[0].get("type") != "sequence":
        return EventPlan(reason="unsupported_temporal_constraints")
    order = constraints[0].get("order")
    if (not isinstance(order, list) or len(order) < 2 or len(actions) != len(events)
            or any(not isinstance(item, str) or not item.strip() for item in order)):
        return EventPlan(reason="invalid_sequence")
    order = [item.strip() for item in order]
    if len(order) > max_events:
        return EventPlan(reason="event_budget_exceeded")
    if (len(set(actions)) != len(actions) or len(set(order)) != len(order)
            or set(order) != set(actions)):
        return EventPlan(reason="ambiguous_or_partial_sequence")
    return EventPlan(tuple(order), "explicit_sequence")
