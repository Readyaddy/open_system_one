"""Bridge between exp7 and Blackjack -- same "describe each option, let the
choice model pick" trick as open_one_agent.py / maze_agent.py.

Unlike Snake and Maze Runner, blackjack has no notion of a "fatal" or
"illegal" option -- hit and stand are both always legal whenever it's the
player's turn, and the risk (busting) is graded, not binary. So the option
text for "hit" carries a bust-probability bucket instead of a safe/unsafe
split, computed by blackjack_env.py's exact card count over every card the
player cannot see (the shoe plus the dealer's hidden hole card -- see that
module's docstring). "stand" carries no risk by definition (you cannot bust
by standing), so its text just states the total being locked in.

A basic-strategy oracle (blackjack_env.basic_strategy_action) scores
whether the model's choice matches textbook play -- shown in the dashboard,
never fed to the model or reflected in the option text.
"""
from blackjack_env import basic_strategy_action

CONTEXT_REQUEST = (
    "The player is playing blackjack and wants to end the hand with a "
    "higher total than the dealer without going over 21."
)
INSTRUCTIONS = "Pick the option that best matches this request."


def _bust_phrase(p: float) -> str:
    if p < 0.15:
        return "very safe, almost never busts"
    if p < 0.35:
        return "fairly safe, a moderate chance of busting"
    if p < 0.60:
        return "risky, a real chance of busting"
    return "very risky, most likely busts"


def render_option(facts) -> str:
    """label -- Category: <description>. -- same rendering family as the
    other two games (data.py's label_description mode + dataset_v7's
    "Category: {x}." template)."""
    soft_tag = " (soft)" if facts.is_soft else ""
    if facts.action == "stand":
        return (f"stand -- Category: locks in the current total of "
                 f"{facts.current_total}{soft_tag}, no risk of busting this turn.")
    return (f"hit -- Category: draws another card on a current total of "
            f"{facts.current_total}{soft_tag}; {_bust_phrase(facts.bust_prob)} "
            f"(bust probability {facts.bust_prob:.0%}).")


def build_request(game):
    """Mirrors the other two games' build_request. Returns (mode, actions,
    option_texts). "model" while it's the player's turn (always exactly 2
    options: hit, stand); "done" once the round has resolved -- nothing to
    decide, the server loop shows the result and starts a new round."""
    if game.phase != "player_turn":
        return "done", [], []
    actions = game.legal_actions()
    option_texts = [render_option(game.move_facts(a)) for a in actions]
    return "model", actions, option_texts


def strategy_match(game, chosen_action: str) -> bool:
    """Whether `chosen_action` matches basic strategy for the CURRENT
    player hand vs. the dealer's up-card -- call this before game.step()
    changes the hand, same as build_request is read before stepping."""
    return basic_strategy_action(game.player, game.dealer[0][0]) == chosen_action
