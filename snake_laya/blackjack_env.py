"""Pure game logic for Blackjack -- no rendering, no model code. Same split
as snake_env.py / maze_env.py: rules and card math live here, the exp7
bridge (blackjack_agent.py) only renders and reads facts this module
computes.

Deliberately the simplest ruleset that is still real blackjack: hit/stand
only (no double, split, or insurance), dealer stands on all 17s (hard or
soft), single fresh 52-card shoe reshuffled every round. Bust probability
on a hit is computed exactly from the cards still unseen to the player
(the full shoe minus every card already dealt face-up AND the dealer's
face-down hole card -- the player genuinely cannot see that card, so it
counts as unseen, exactly matching what a real player would be reasoning
from) -- not simulated, counted directly, since the deck is small enough
that exact counting is both cheap and unambiguous.
"""
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["S", "H", "D", "C"]


def rank_value(rank: str) -> int:
    if rank in ("J", "Q", "K"):
        return 10
    if rank == "A":
        return 11  # soft value; hand_value() below demotes aces to 1 as needed
    return int(rank)


def fresh_deck(rng: random.Random) -> List[Tuple[str, str]]:
    deck = [(r, s) for r in RANKS for s in SUITS]
    rng.shuffle(deck)
    return deck


def hand_value(cards: List[Tuple[str, str]]) -> Tuple[int, bool]:
    """Returns (total, is_soft). is_soft is True iff at least one ace is
    still being counted as 11 -- standard blackjack soft-hand definition."""
    total = sum(rank_value(r) for r, _ in cards)
    aces = sum(1 for r, _ in cards if r == "A")
    soft = aces > 0
    while total > 21 and aces > 0:
        total -= 10  # demote one ace from 11 to 1
        aces -= 1
        soft = aces > 0
    return total, soft


def is_blackjack(cards: List[Tuple[str, str]]) -> bool:
    return len(cards) == 2 and hand_value(cards)[0] == 21


@dataclass
class MoveFacts:
    """Everything true about one candidate action ("hit" or "stand"),
    computed from information the player can actually see -- the dealer's
    hole card is never used here, matching what a real player reasons from.
    """
    action: str
    bust_prob: Optional[float]  # None for "stand" -- busting isn't possible by standing
    current_total: int   # the hand's total right now, before this action -- same
    # number for both "hit" and "stand" facts of the same turn (it describes the
    # hand, not a hypothetical outcome; an earlier version tried to report a
    # "total if the hit is safe" estimate here, which was confusing on screen and
    # not even used for the real bust_prob math -- current_total is simpler and
    # is exactly what a player looks at before deciding)
    is_soft: bool
    dealer_upcard_value: int


@dataclass
class BlackjackGame:
    rng: random.Random = field(default_factory=random.Random)
    deck: List[Tuple[str, str]] = field(default_factory=list)
    player: List[Tuple[str, str]] = field(default_factory=list)
    dealer: List[Tuple[str, str]] = field(default_factory=list)
    phase: str = "player_turn"   # "player_turn" | "dealer_turn" | "done"
    result: Optional[str] = None  # "player_bust" | "dealer_bust" | "player_win" |
    # "dealer_win" | "push" | "player_blackjack" -- set once phase == "done"
    rounds_played: int = 0
    rounds_won: int = 0

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.deck = fresh_deck(self.rng)
        self.player = [self.deck.pop(), self.deck.pop()]
        self.dealer = [self.deck.pop(), self.deck.pop()]
        self.phase = "player_turn"
        self.result = None

        # Natural blackjack resolves immediately, before the player gets a
        # hit/stand choice -- standard rule, and the reason this check has
        # to happen in reset() rather than only in step().
        player_bj, dealer_bj = is_blackjack(self.player), is_blackjack(self.dealer)
        if player_bj or dealer_bj:
            self.phase = "done"
            if player_bj and dealer_bj:
                self.result = "push"
            elif player_bj:
                self.result = "player_blackjack"
            else:
                self.result = "dealer_win"
            self._settle()
        return self

    def legal_actions(self):
        return ["hit", "stand"] if self.phase == "player_turn" else []

    def _unseen_cards(self):
        """Every deck card not visible to the player: the remaining shoe
        plus the dealer's hole card (dealer[1]) -- the player cannot see
        that card, so bust-probability math must not assume knowledge of
        it. dealer[0] IS visible (it's the up-card) and player's own cards
        are obviously visible, so neither is included here."""
        return list(self.deck) + [self.dealer[1]]

    def move_facts(self, action: str) -> MoveFacts:
        total, soft = hand_value(self.player)
        dealer_up = rank_value(self.dealer[0][0])  # dealer[0] is a (rank, suit) tuple
        if action == "stand":
            return MoveFacts(action="stand", bust_prob=None, current_total=total,
                              is_soft=soft, dealer_upcard_value=dealer_up)

        unseen = self._unseen_cards()
        bust_count = sum(1 for r, _ in unseen if hand_value(self.player + [(r, "?")])[0] > 21)
        bust_prob = bust_count / len(unseen) if unseen else 0.0
        return MoveFacts(action="hit", bust_prob=bust_prob, current_total=total,
                          is_soft=soft, dealer_upcard_value=dealer_up)

    def step(self, action: str):
        if self.phase != "player_turn":
            return self.phase
        if action == "hit":
            self.player.append(self.deck.pop())
            total, _ = hand_value(self.player)
            if total > 21:
                self.phase = "done"
                self.result = "player_bust"
                self._settle()
        elif action == "stand":
            self._play_dealer()
        return self.phase

    def _play_dealer(self):
        self.phase = "dealer_turn"
        while True:
            total, _ = hand_value(self.dealer)
            if total >= 17:
                break
            self.dealer.append(self.deck.pop())
        dealer_total, _ = hand_value(self.dealer)
        player_total, _ = hand_value(self.player)
        if dealer_total > 21:
            self.result = "dealer_bust"
        elif dealer_total > player_total:
            self.result = "dealer_win"
        elif dealer_total < player_total:
            self.result = "player_win"
        else:
            self.result = "push"
        self.phase = "done"
        self._settle()

    def _settle(self):
        self.rounds_played += 1
        if self.result in ("player_win", "dealer_bust", "player_blackjack"):
            self.rounds_won += 1

    def describe(self):
        pt, psoft = hand_value(self.player)
        return (f"Player: {self.player} (total {pt}{' soft' if psoft else ''}) | "
                f"Dealer shows: {self.dealer[0]} | phase={self.phase} result={self.result}")


# --------------------------------------------------------------------------
# Basic strategy oracle -- used ONLY to score whether the model's hit/stand
# choice matches textbook basic strategy (a "did it play well" diagnostic,
# same role Snake's BFS/flood-fill oracle plays), never shown to the model
# and never used to build its option text. Simplified to hit/stand only (no
# double/split, since the game doesn't offer those actions) -- standard
# basic-strategy charts collapse to this once double/split are removed:
# hit on anything below the stand thresholds below, stand otherwise.
# --------------------------------------------------------------------------

def basic_strategy_action(player_cards, dealer_upcard_rank) -> str:
    total, soft = hand_value(player_cards)
    up = rank_value(dealer_upcard_rank)
    if soft:
        # Soft totals: stand on soft 19+ always; soft 18 stands vs. dealer
        # 2,7,8, hits otherwise; below that, always hit (doubling is the
        # textbook action on soft 13-18 vs. weak dealer cards, but this
        # ruleset has no double, so it collapses to hit).
        if total >= 19:
            return "stand"
        if total == 18:
            return "stand" if up in (2, 7, 8) else "hit"
        return "hit"
    # Hard totals.
    if total >= 17:
        return "stand"
    if total <= 11:
        return "hit"
    if total == 12:
        return "stand" if up in (4, 5, 6) else "hit"
    # 13-16: stand only against a weak dealer up-card (2-6), hit otherwise.
    return "stand" if up in (2, 3, 4, 5, 6) else "hit"
