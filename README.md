# Checkers 6x6

A two-player 6×6 Checkers environment following the [PettingZoo AEC API](https://pettingzoo.farama.org/api/aec/).

---

## Environment Description

A simplified version of Checkers played on a 6×6 board. Two agents take turns moving pieces diagonally. Capturing is mandatory. A piece that reaches the opponent's back row becomes a King.

---

## Usage

```python
from mycheckersenv import CheckersEnv

env = CheckersEnv()
env.reset()

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        env.step(None)
    else:
        mask   = obs["action_mask"]
        action = env.action_space(agent).sample(mask)
        env.step(action)

env.close()
```

---

## Agents

| Agent      | Description          |
|------------|----------------------|
| `player_1` | Pieces start rows 4–5, move toward row 0 |
| `player_2` | Pieces start rows 0–1, move toward row 5 |

---

## Observation Space

Each agent receives a dictionary:

```
{
  "observation": Box(shape=(6, 6), dtype=int8, low=-2, high=2),
  "action_mask": MultiBinary(1296)
}
```

### Observation values

| Value | Meaning           |
|-------|-------------------|
| `0`   | Empty square      |
| `1`   | player_1 man      |
| `2`   | player_1 king     |
| `-1`  | player_2 man      |
| `-2`  | player_2 king     |

Only dark squares (where `(row + col) % 2 == 1`) are ever occupied.

### Action Mask

A binary vector of length **1296** (= 36 × 36). Entry `i` is `1` if action `i` is legal, `0` otherwise. If any capture is available, only captures are marked legal (mandatory capture rule).

---

## Action Space

`Discrete(1296)`

An action is an integer encoding a `(from_square, to_square)` pair:

```
action    = from_pos * 36 + to_pos
from_pos  = from_row * 6 + from_col
to_pos    = to_row   * 6 + to_col
```

Valid actions are diagonal steps of 1 square (simple move) or 2 squares (capture jump). Use the `action_mask` to filter legal actions.

---

## Rewards

| Event                              | Reward for winner | Reward for loser |
|------------------------------------|:-----------------:|:----------------:|
| Opponent has no pieces remaining   | `+1`              | `-1`             |
| Opponent has no legal moves        | `+1`              | `-1`             |
| Illegal move submitted             | `+1`              | `-1`             |
| All other steps                    | `0`               | `0`              |

If `max_cycles` is reached the episode is truncated with reward `0` for both agents.

---

## Termination / Truncation

| Condition                              | Type        |
|----------------------------------------|-------------|
| A player loses all their pieces        | Termination |
| A player has no legal moves            | Termination |
| A player submits an illegal action     | Termination |
| `max_cycles` steps reached (default 200) | Truncation |

---

## Game Rules Summary

- Pieces move diagonally on dark squares only.
- **Men** move forward only (toward the opponent's side).
- **Kings** (promoted men) may move in all four diagonal directions.
- **Capturing is mandatory**: if a jump is available, a non-jumping move is not legal.
- A man reaching the opponent's back row is promoted to a King immediately.
- Win by capturing all opponent pieces or leaving them with no legal moves.

---

## Parameters

| Parameter    | Default | Description                       |
|--------------|---------|-----------------------------------|
| `max_cycles` | `200`   | Maximum steps before truncation   |
