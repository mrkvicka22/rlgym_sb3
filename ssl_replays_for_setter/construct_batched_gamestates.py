import pickle

from rlgym_tools.replay_converter import convert_replay
from os import listdir
from os.path import join
from rlgym.utils.gamestates import GameState
from typing import List
from numpy.random import shuffle


def write_batch(batch: List[GameState], game_mode: str, number: int):
    with open(f"batched_gamestates/{game_mode}/{str(number)}.gamestate", "wb") as f:
        pickle.dump(batch, f)


if __name__ == '__main__':
    path_to_replays = 'replays'
    frame_skip = 15
    batch_size = 512
    for game_mode in listdir(path_to_replays):
        states = []
        for replay in listdir(join(path_to_replays, game_mode)):

            replay_iterator = convert_replay(join(path_to_replays, game_mode, replay))
            for i, value in enumerate(replay_iterator):
                if i % frame_skip == frame_skip - 1:
                    state, _ = value
                    states.append(state)
        shuffle(states)
        batches = [states[i:i + batch_size] for i in range(0, len(states), batch_size)]
        for i, batch in enumerate(batches):
            write_batch(batch, game_mode, i)
            print('wrote one')
        print(f'finished game mode {game_mode}')
