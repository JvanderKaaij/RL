import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    window_size = 512

    def __init__(self, columns=7, rows=6, connections=4, render_mode=None):
        self.columns = columns
        self.rows = rows
        self.connections = connections
        self.current_player = 0
        self.winner = None
        self.board = np.full((self.columns, self.rows), -1)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(columns, rows), dtype=np.int32)
        self.action_space = spaces.Discrete(columns)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self.board

    def _get_info(self):
        return {
            'board': self.board,
            'current_player': self.current_player
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0
        self.board = np.full((self.columns, self.rows), -1)

        observation = self._get_obs()
        info = self._get_info()
        self.winner = None

        return observation, info

    def step(self, action):
        move_column = action
        # punish player for an action where the top row in column is already full
        if not self.board[move_column][self.rows - 1] == -1:
            reward = -1  # TODO skip rest OR MOVE TO CHECK FOR EPISODE TERMINATION

        row = self.rows - 1
        while 0 < row < self.rows and self.board[move_column][row] != -1:
            row -= 1

        self.board[move_column][row] = self.current_player

        self.winner, reward_vector = self.check_for_episode_termination(move_column, row)

        reward = 0

        terminated = self.winner is not None

        observation = self._get_obs()
        info = self._get_info()
        self._switch_player()
        # TODO Not sure yet how to deal with reward vector here, as I'm unsure how the training should work
        return observation, reward, terminated, False, info

    def check_for_episode_termination(self, movecol, row):
        winner, reward_vector = self.winner, [0, 0]
        if self._does_move_win(movecol, row):
            winner = self.current_player
            if winner == 0:
                reward_vector = [1, -1]
            elif winner == 1:
                reward_vector = [-1, 1]
        elif not self._get_moves():  # A draw has happened
            winner = -1
            reward_vector = [0, 0]
        return winner, reward_vector

    def _get_moves(self):
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        if self.winner is not None:
            return []
        return [col for col in range(self.columns) if self.board[col][self.rows - 1] == -1]

    def _switch_player(self):
        self.current_player = (self.current_player + 1) % 2

    def _is_on_board(self, x, y):
        return 0 <= x < self.columns and 0 <= y < self.rows

    def _does_move_win(self, x, y):
        """
        Checks whether a newly dropped chip at position param x, param y
        wins the game.
        :param x: column index
        :param y: row index
        :returns: (boolean) True if the previous move has won the game
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1
            while self._is_on_board(x + p * dx, y + p * dy) and self.board[x + p * dx][y + p * dy] == self.current_player:
                p += 1
            n = 1
            while self._is_on_board(x - n * dx, y - n * dy) and self.board[x - n * dx][y - n * dy] == self.current_player:
                n += 1

            if p + n >= (self.connections + 1):  # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >- 5
                return True

        return False
