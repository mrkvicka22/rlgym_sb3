from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from numpy import ndarray


class KeepAwayReward(RewardFunction):
    """
    Rewards for every tick that player's team has had the last touch.
    """

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        # check for touch occuring
        if state.last_touch == -1:
            return 0
        if player.team_num == state.players[state.last_touch].team_num:
            return 1
