import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces

class Checkers6x6Env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "checkers_6x6_v0"}

    def __init__(self):
        super().__init__()

        self.board_size = 6
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Observation: 6x6 board
        self.observation_spaces = {
            agent: spaces.Box(low=-2, high=2, shape=(6, 6), dtype=np.int8)
            for agent in self.agents
        }

        # Action: (from_row, from_col, to_row, to_col)
        self.action_spaces = {
            agent: spaces.MultiDiscrete([6, 6, 6, 6])
            for agent in self.agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.board = self._init_board()

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.agent_selection = self._agent_selector.reset()

    def _init_board(self):
        board = np.zeros((6, 6), dtype=np.int8)

        # Player 1 (top)
        for r in range(2):
            for c in range(6):
                if (r + c) % 2 == 1:
                    board[r][c] = -1

        # Player 0 (bottom)
        for r in range(4, 6):
            for c in range(6):
                if (r + c) % 2 == 1:
                    board[r][c] = 1

        return board

    def observe(self, agent):
        return self.board.copy()

    def step(self, action):
        if self.dones[self.agent_selection]:
            self._was_done_step(action)
            return

        agent = self.agent_selection
        player = 1 if agent == "player_0" else -1

        fr, fc, tr, tc = action

        valid_moves = self._get_legal_moves(player)

        if (fr, fc, tr, tc) not in valid_moves:
            # illegal move penalty
            self.rewards[agent] = -1
            self._next_agent()
            return

        self._apply_move(fr, fc, tr, tc)

        # Check win condition
        if self._check_win(player):
            self.rewards[agent] = 1
            opponent = self.agents[1 - self.agents.index(agent)]
            self.rewards[opponent] = -1
            self.dones = {a: True for a in self.agents}

        self._next_agent()

    def _next_agent(self):
        self.agent_selection = self._agent_selector.next()

    def _apply_move(self, fr, fc, tr, tc):
        piece = self.board[fr][fc]
        self.board[fr][fc] = 0
        self.board[tr][tc] = piece

        # Capture
        if abs(fr - tr) == 2:
            mid_r = (fr + tr) // 2
            mid_c = (fc + tc) // 2
            self.board[mid_r][mid_c] = 0

        # King promotion
        if piece == 1 and tr == 0:
            self.board[tr][tc] = 2
        elif piece == -1 and tr == 5:
            self.board[tr][tc] = -2

    def _get_legal_moves(self, player):
        moves = []
        captures = []

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(6):
            for c in range(6):
                piece = self.board[r][c]

                if piece * player <= 0:
                    continue

                is_king = abs(piece) == 2

                for dr, dc in directions:
                    if not is_king:
                        if player == 1 and dr > 0:
                            continue
                        if player == -1 and dr < 0:
                            continue

                    # Normal move
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 6 and 0 <= nc < 6:
                        if self.board[nr][nc] == 0:
                            moves.append((r, c, nr, nc))

                    # Capture
                    nr2, nc2 = r + 2*dr, c + 2*dc
                    if 0 <= nr2 < 6 and 0 <= nc2 < 6:
                        mid = self.board[r + dr][c + dc]
                        if mid * player < 0 and self.board[nr2][nc2] == 0:
                            captures.append((r, c, nr2, nc2))

        return captures if captures else moves  # mandatory capture

    def _check_win(self, player):
        opponent = -player

        # No opponent pieces
        if not np.any(self.board * opponent > 0):
            return True

        # No legal moves
        if len(self._get_legal_moves(opponent)) == 0:
            return True

        return False

    def render(self):
        print(self.board)