"""Roles, scopes, tiers and the agent each role gets. This is the single place that decides
which tools an operator, a service engineer, a safety lead and remote support can use.

Only the operator is defined so far (task 11). The other roles arrive with tasks 12 and 13.
"""
from __future__ import annotations

from .agent.guards import grounding_check
from .agent.loop import AgentConfig
from .tools.registry import Principal

ROLE_SCOPES: dict[str, frozenset[str]] = {
    "operator": frozenset({"robot:read", "robot:move", "episodes:read"}),
}
ROLE_TIERS: dict[str, frozenset[str]] = {
    "operator": frozenset({"observe", "simulate", "actuate"}),
}


def principal(user_id: str, role: str, site: str, display_name: str = "") -> Principal:
    return Principal(user_id, role, ROLE_SCOPES[role], site=site, tiers=ROLE_TIERS[role],
                     display_name=display_name or user_id)


DEMO = ("This is a research demonstration. The robot is a kinematic twin of a dVRK arm: it has no contact, no "
        "needle and no camera, and it is not connected to a real robot.")

OPERATOR = AgentConfig(
    name="operator_copilot", max_iterations=12, token_budget=60000, answer_check=grounding_check,
    tool_names=["get_robot_state", "list_named_poses", "check_named_target", "rehearse_move", "enable_and_home",
                "move_to_named_pose", "set_jaw", "wait_action", "cancel_action", "list_episodes",
                "episode_summary", "read_signal_window", "calculator"],
    system_prompt=(
        "You assist the operator of a da Vinci Research Kit (dVRK) arm. " + DEMO + "\n"
        "Rules:\n"
        "1. Look before you speak. Anything you say about the robot must come from a tool result in this "
        "conversation.\n"
        "2. Numbers come from tools, with their unit. Use calculator for any arithmetic. Never estimate.\n"
        "3. You cannot send coordinates. The arm only goes to poses named by list_named_poses.\n"
        "4. Before any motion call get_robot_state and pass its state_seq as based_on_seq. If a motion is refused "
        "as stale_observation, look again and make the request again.\n"
        "5. Send one motion at a time, as the only tool call in your response, and read its result before "
        "deciding anything else.\n"
        "6. If the safety filter rejects a target, say which rule rejected it and stop. Never look for another "
        "way to reach a rejected target.\n"
        "7. If a motion is aborted or canceled, do not repeat it. Report where the arm is.\n"
        "8. If the arm reports a fault, do not command it. Report the fault.\n"
        "9. If a tool fails, say what failed and continue with what you have. Do not invent the missing result.\n"
        "10. check_named_target and rehearse_move do not move the arm. When you are asked to move, carry out "
        "every requested motion with a motion tool, in the order asked, unless the safety filter does not accept "
        "it. If you did not move the arm, say so plainly.\n"
        "Be concise and specific."))

AGENTS = {"operator": OPERATOR}
