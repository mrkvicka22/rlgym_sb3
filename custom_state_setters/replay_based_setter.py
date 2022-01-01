import os

from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.state_setters.wrappers import PhysicsWrapper, CarWrapper
from rlgym.utils.gamestates.game_state import GameState
import numpy as np
import pickle


class RandomSSLSetter(StateSetter):

    def __init__(self, path_to_batches: str, shuffle_batches: bool = False):
        """
        RandomState constructor.


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
        # state_wrapper.ball = PhysicsWrapper(game_state.ball)
        # state_wrapper.cars = []
        # for player in game_state.players:
        #     state_wrapper.cars.append(CarWrapper(player_data=player))

        self._set_ball(state_wrapper, game_state)
        self._set_cars(state_wrapper, game_state)
        self.batch.pop()

    def _set_cars(self, state_wrapper: StateWrapper, game_state: GameState):
        for car, player in zip(state_wrapper.cars, game_state.players):
            car.set_pos(*player.car_data.position)
            car.set_rot(pitch=player.car_data.pitch(), yaw=player.car_data.yaw(), roll=player.car_data.roll())
            car.set_ang_vel(*player.car_data.angular_velocity)
            car.set_lin_vel(*player.car_data.linear_velocity)
            car.boost = player.boost_amount

    def _set_ball(self, state_wrapper: StateWrapper, game_state: GameState):
        state_wrapper.ball.set_pos(*game_state.ball.position)
        state_wrapper.ball.set_ang_vel(*game_state.ball.angular_velocity)
        state_wrapper.ball.set_lin_vel(*game_state.ball.linear_velocity)

    def _load_batch(self, team_size: int):
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
        except StopIteration:
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
