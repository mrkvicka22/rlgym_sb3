import os

from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.gamestates.game_state import GameState
from typing import List, Union
import numpy as np
import pickle
from ssl_replays_for_setter.construct_batched_gamestates import convert_replays
from rlgym_tools.replay_converter import convert_replay


class ReplayBasedSetter(StateSetter):

    def __init__(self, path_to_batches: str, shuffle_batches: bool = False):
        """
        ReplayBasedSetter constructor.

        :param path_to_batches: Path to the directory that has
        :param shuffle_batches: Boolean indicating whether to shuffle batches or go through them in alphabetical order.
        """
        super().__init__()
        self.path_to_batches = path_to_batches

        self.threes_file_names = os.listdir(os.path.join(self.path_to_batches, '3'))
        self.twos_file_names = os.listdir(os.path.join(self.path_to_batches, '2'))
        self.ones_file_names = os.listdir(os.path.join(self.path_to_batches, '1'))
        if shuffle_batches:
            np.random.shuffle(self.threes_file_names)
            np.random.shuffle(self.twos_file_names)
            np.random.shuffle(self.ones_file_names)

        self.threes_file_names_iter = iter(self.threes_file_names)
        self.twos_file_names_iter = iter(self.twos_file_names)
        self.ones_file_names_iter = iter(self.ones_file_names)

        self.batch = None

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        if not self.batch:
            self.batch = self._load_batch(len(state_wrapper.cars) // 2)
        game_state = self.batch[-1]

        self._set_ball(state_wrapper, game_state)
        self._set_cars(state_wrapper, game_state)
        self.batch.pop()

    def _set_cars(self, state_wrapper: StateWrapper, game_state: GameState):
        """
        Sets the players according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param game_state: GameState object from the replay to get data from.
        """
        for car, player in zip(state_wrapper.cars, game_state.players):
            car.set_pos(*player.car_data.position)
            car.set_rot(pitch=player.car_data.pitch(), yaw=player.car_data.yaw(), roll=player.car_data.roll())
            car.set_ang_vel(*player.car_data.angular_velocity)
            car.set_lin_vel(*player.car_data.linear_velocity)
            car.boost = player.boost_amount

    def _set_ball(self, state_wrapper: StateWrapper, game_state: GameState):
        """
        Sets the ball according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param game_state: GameState object from the replay to get data from.
        """
        state_wrapper.ball.set_pos(*game_state.ball.position)
        state_wrapper.ball.set_ang_vel(*game_state.ball.angular_velocity)
        state_wrapper.ball.set_lin_vel(*game_state.ball.linear_velocity)

    def _load_batch(self, team_size: int) -> List[GameState]:
        """
        Handles getting next batch for the appropriate game mode.

        :param team_size: How many players should each team have
        :return: List of GameStates
        """
        # handles getting the next batch for the appropriate game mode.
        try:
            if team_size == 1:
                batch_name = next(self.ones_file_names_iter)
            elif team_size == 2:
                batch_name = next(self.twos_file_names_iter)
            elif team_size == 3:
                batch_name = next(self.threes_file_names_iter)
            else:
                raise NotImplementedError(
                    f"team_size other than 1,2 or 3 are not supported. However, team_size {team_size} was given.")
        except StopIteration:  # this can happen if we run out of batches. Should be very rare.
            self.threes_file_names_iter = iter(self.threes_file_names)
            self.twos_file_names_iter = iter(self.twos_file_names)
            self.ones_file_names_iter = iter(self.ones_file_names)
            if team_size == 1:
                batch_name = next(self.ones_file_names_iter)
            elif team_size == 2:
                batch_name = next(self.twos_file_names_iter)
            elif team_size == 3:
                batch_name = next(self.threes_file_names_iter)
            else:
                raise NotImplementedError(
                    f"team_size other than 1,2 or 3 are not supported. However, team_size {team_size} was given.")

        with open(os.path.join(self.path_to_batches, str(team_size), batch_name), "rb") as f:
            new_batch = pickle.load(f)
        return new_batch


class ReplayBasedSetterv2(StateSetter):

    def __init__(self, ndarray_or_file: Union[str, np.ndarray]):
        """
        ReplayBasedSetter constructor

        :param states_list_or_file: A file string or a list of states for each game mode.
        """
        super().__init__()

        if isinstance(ndarray_or_file, np.ndarray):
            self.states = ndarray_or_file
        elif isinstance(ndarray_or_file, str):
            with open(ndarray_or_file, "rb") as f:
                self.states = pickle.load(f)

    @classmethod
    def construct_from_replays(cls, replays, frame_skip: int = 150):
        return cls(convert_replays(replays, frame_skip))

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to contain random values the ball and each car.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """

        data = self.states[np.random.randint(0, len(self.states))]
        assert len(data) == len(state_wrapper.cars)*13 + 9, "Data given does not match current game mode"
        self._set_ball(state_wrapper, data)
        self._set_cars(state_wrapper, data)

    def _set_cars(self, state_wrapper: StateWrapper, data: np.ndarray):
        """
        Sets the players according to the game state from replay

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        :param game_state: GameState object from the replay to get data from.
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
        :param game_state: GameState object from the replay to get data from.
        """
        state_wrapper.ball.set_pos(*data[:3])
        state_wrapper.ball.set_lin_vel(*data[3:6])
        state_wrapper.ball.set_ang_vel(*data[6:9])
