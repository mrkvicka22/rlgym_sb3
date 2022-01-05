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
    states = []
    for replay in paths_to_each_replay:
        make_state_data(replay, states, frame_skip)
        if verbose > 0:
            print(replay, "done")

    return np.asarray(states)


def make_state_data(replay, states, frame_skip):
    replay_iterator = convert_replay(replay)
    for i, value in enumerate(replay_iterator):
        if i % frame_skip == frame_skip - 1:
            state, _ = value
            np_state = state_to_np_array(state)
            states.append(np_state)


def state_to_np_array(game_state: GameState) -> np.ndarray:
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
