import os
import pickle

import numpy as np
from rlgym_tools.replay_converter import convert_replay
from rlgym.utils.gamestates import GameState
from typing import List


def absolute_file_paths(directory: str):
    for dir_path, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dir_path, f))


def convert_replays(paths_to_each_replay: List[str], frame_skip: int = 150, verbose: int = 0) -> np.ndarray:
    """
    Converts replays into

    :param paths_to_each_replay: List of paths to each replay to be converted
    :param frame_skip: Every frame_skip frame from the replay will be converted
    :param verbose: When set to more than 0, it prints debug info
    :return: Numpy array of frames
    """
    states = []
    for replay in paths_to_each_replay:
        make_state_data(replay, states, frame_skip)
        if verbose > 0:
            print(replay, "done")

    return np.asarray(states)


def make_state_data(replay: str, states: list, frame_skip: int):
    """
    Process single replay

    :param replay: Path to the replay to process
    :param states: List of all the processed states
    :param frame_skip: Every frame_skip frame from the replay will be converted
    """
    replay_iterator = convert_replay(replay)
    for i, value in enumerate(replay_iterator):
        if i % frame_skip == frame_skip - 1:
            state, _ = value
            np_state = state_to_np_array(state)
            states.append(np_state)


def state_to_np_array(game_state: GameState) -> np.ndarray:
    """
    Gets only the useful information from game_state and puts it into numpy array

    :param game_state: GameState to convert
    :return: Numpy array of the most useful information from the game_state
    """
    whole_state = []
    ball = game_state.ball
    ball_state = np.concatenate((ball.position, ball.linear_velocity, ball.angular_velocity))

    whole_state.append(ball_state)
    for player in game_state.players:
        whole_state.append(np.concatenate((player.car_data.position,
                                           player.car_data.euler_angles(),
                                           player.car_data.linear_velocity,
                                           player.car_data.angular_velocity,
                                           np.asarray([player.boost_amount]))))
    return np.concatenate(whole_state)


def main():
    replay_names = list(absolute_file_paths("replays/3"))
    converted_states = convert_replays(replay_names, verbose=1)
    print(converted_states[10])
    with open("saved_gamestates3.gamestate", "wb") as f:
        pickle.dump(converted_states, f)


if __name__ == '__main__':
    main()
