import os

from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.gamestates.game_state import GameState
from typing import List, Union
import numpy as np
import pickle
from ssl_replays_for_setter.construct_batched_gamestates import convert_replays


class ReplayBasedSetter(StateSetter):

    def __init__(self, ndarray_or_file: Union[str, np.ndarray]):
        """
        ReplayBasedSetter constructor

        :param ndarray_or_file: A file string or a numpy ndarray of states for a single game mode.
        """
        super().__init__()

        if isinstance(ndarray_or_file, np.ndarray):
            self.states = ndarray_or_file
        elif isinstance(ndarray_or_file, str):
            with open(ndarray_or_file, "rb") as f:
                self.states = pickle.load(f)

    @classmethod
    def construct_from_replays(cls, paths_to_replays: List[str], frame_skip: int = 150):
        """
        Alternative constructor that constructs ReplayBasedSetter from replays given as paths.

        :param paths_to_replays: Paths to all the reapls
        :param frame_skip: Every frame_skip frame from the replay will be converted
        :return: Numpy array of frames
        """
        return cls(convert_replays(paths_to_replays, frame_skip))

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        data = self.states[np.random.randint(0, len(self.states))]
        assert len(data) == len(state_wrapper.cars) * 13 + 9, "Data given does not match current game mode"
        self._set_ball(state_wrapper, data)
        self._set_cars(state_wrapper, data)

    def _set_cars(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the players according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """

        data = np.split(data[9:], len(state_wrapper.cars))
        for i, car in enumerate(state_wrapper.cars):
            car.set_pos(*data[i][:3])
            car.set_rot(pitch=data[i][3], yaw=data[i][4], roll=data[i][5])
            car.set_lin_vel(*data[i][6:9])
            car.set_ang_vel(*data[i][9:12])
            car.boost = data[i][12]

    def _set_ball(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the ball according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param data: Numpy array from the replay to get values from.
        """
        state_wrapper.ball.set_pos(*data[:3])
        state_wrapper.ball.set_lin_vel(*data[3:6])
        state_wrapper.ball.set_ang_vel(*data[6:9])
