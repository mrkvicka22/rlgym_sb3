import rlgym_compat
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from numpy import ndarray
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM


class KeepAwayReward(RewardFunction):
    """
    Rewards for every tick that player's team has had the last touch.
    """

    def __init__(self):
        super().__init__()
        self.player_ids = {BLUE_TEAM:{1,2,3,4},ORANGE_TEAM:{5,6,7,8}}

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        # check for touch occuring
        if state.last_touch == -1:
            return 0
        if state.last_touch in self.player_ids[player.team_num]:
            return 1
        else:
            return 0
