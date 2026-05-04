"""Baseline trading policies + PnL math used by run_baselines.py.

A *policy* takes a market row (dict-like) and the calibration table, and
returns one of:

    None                        -> HOLD (no trade)
    ("YES", contracts)          -> BUY YES contracts at yes_ask
    ("NO",  contracts)          -> BUY NO  contracts at no_ask

PnL is computed in dollars per contract. For a BUY at price p (cents) on a
market that resolves the same side, payoff = $1.00 - $p/100; otherwise
payoff = -$p/100.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

# Type aliases
Action = Optional[tuple[str, float]]
Policy = Callable[[pd.Series, dict[int, float]], Action]


# --- pnl ---------------------------------------------------------------------


def pnl_per_contract(side: str, price_cents: int, result: str) -> float:
    """$ profit per 1 contract for a BUY of ``side`` at ``price_cents``."""
    cost = price_cents / 100.0
    win = (side == "YES" and result == "yes") or (side == "NO" and result == "no")
    return (1.0 - cost) if win else -cost


def episode_pnl(action: Action, market: pd.Series) -> float:
    if action is None:
        return 0.0
    side, contracts = action
    price_cents = int(market["yes_ask"]) if side == "YES" else int(market["no_ask"])
    return contracts * pnl_per_contract(side, price_cents, market["result"])


# --- policies ----------------------------------------------------------------


def policy_hold(_market: pd.Series, _calib: dict[int, float]) -> Action:
    return None


def policy_always_yes(market: pd.Series, _calib: dict[int, float]) -> Action:
    return ("YES", 1.0)


def policy_always_no(market: pd.Series, _calib: dict[int, float]) -> Action:
    return ("NO", 1.0)


def policy_price_as_prob(market: pd.Series, _calib: dict[int, float]) -> Action:
    """Treat yes_ask/100 as P(yes). BUY whichever side is cheaper than its prob.

    Equivalent to: assume the market is calibrated; only bet to break ties
    when one side is strictly underpriced relative to that assumption.
    """
    p = int(market["yes_ask"]) / 100.0
    if int(market["no_ask"]) / 100.0 < (1 - p):
        return ("NO", 1.0)
    if p < p:  # never true -- price-as-prob => zero edge
        return ("YES", 1.0)
    return None


def policy_ev_greedy(market: pd.Series, calib: dict[int, float]) -> Action:
    """Use empirical calibration table to pick the side with positive EV.

    EV(YES) = win_rate * (1 - p) - (1 - win_rate) * p   (in $/contract)
    EV(NO)  = (1 - win_rate) * (1 - q) - win_rate * q    where q = no_ask/100
    """
    p_cents = int(market["yes_ask"])
    q_cents = int(market["no_ask"])
    win_rate = calib.get(p_cents, p_cents / 100.0)
    p = p_cents / 100.0
    q = q_cents / 100.0
    ev_yes = win_rate * (1 - p) - (1 - win_rate) * p
    ev_no = (1 - win_rate) * (1 - q) - win_rate * q
    if max(ev_yes, ev_no) <= 0:
        return None
    return ("YES", 1.0) if ev_yes >= ev_no else ("NO", 1.0)


POLICIES: dict[str, Policy] = {
    "hold": policy_hold,
    "always_yes": policy_always_yes,
    "always_no": policy_always_no,
    "price_as_prob": policy_price_as_prob,
    "ev_greedy": policy_ev_greedy,
}
