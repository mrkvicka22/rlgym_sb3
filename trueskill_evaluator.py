import os
import rlgym
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from trueskill import Rating, rate_1vs1, global_env, setup
import pickle
from rlgym.utils.state_setters.default_state import DefaultState
from stable_baselines3.ppo import PPO
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, \
    NoTouchTimeoutCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
#from state_setters.symmetric_rand_state import SymmetricRandState

# Disable cpu parallelization
torch.set_num_threads(1)

# Setup TrueSkill env
setup(draw_probability=0.01)
ts = global_env()


def get_policies():
    return list(sorted(os.listdir("policy"), key=lambda x: int(x.split("_")[2])))

def get_opponent_in_range(ratings: dict, min_mu, max_mu):
    # Get MUs
    mus = np.asarray([ratings[i]["rating"].mu for i in range(len(ratings)-1)])

    # Get opponents in rating range
    op_indexes = np.where((mus >= min_mu) & (mus <= max_mu))[0]

    # If op_indexes.length == 0, get the closest opponent
    if len(op_indexes) == 0:
        # Get the closest opponent
        print("No opponents in mmr range, getting closest agent")
        mus_distance = np.abs(mus - (min_mu + max_mu) / 2)
        index = int(np.argmin(mus_distance))
    else:
        # Choose one
        index = int(np.random.choice(op_indexes))
    return ratings[index]
# Initialize rating list if necessary (length == 0)
def initialize_ratings(reset_ratings: bool = False):
    try:
        with open("policy_ratings", "rb") as f:
            ratings = pickle.load(f)
    except FileNotFoundError:
        ratings = []
        for model in get_policies():
            ratings.append({"name":model, "rating":Rating()})
    if reset_ratings:
        ratings = []
        for model in get_policies():
            ratings.append({"name":model, "rating":Rating()})
    return ratings


# Initialize rlgym
team_size = 2
max_steps = 3000
no_touch_steps = 500
action_trans = np.array([-1, -1, -1, -1, -1, 0, 0, 0])
env = rlgym.make(team_size=team_size, self_play=True, use_injector=True,
                 obs_builder=AdvancedObs(),
                 state_setter=DefaultState(),
                 terminal_conditions=[TimeoutCondition(max_steps), GoalScoredCondition(),
                                      NoTouchTimeoutCondition(no_touch_steps)],
                 action_parser=KBMAction())
# agent = PPOActor(state_space=env.observation_space.shape[0], action_categoricals=5, action_bernoullis=3)
# opponent = PPOActor(state_space=env.observation_space.shape[0], action_categoricals=5, action_bernoullis=3)

while True:
    # Get reference Mu -> Returns last MU and last index
    ratings = initialize_ratings()
    agent_idx = -1
    # initial_mu = float(redis.lindex(Keys.OP_MUS, -1))
    initial_mu = ratings[-1]["rating"].mu
    agent_rating = Rating(mu=initial_mu)

    # TODO remove plotting
    mus = [agent_rating.mu]

    agent = PPO.load('policy/'+get_policies()[-1])

    # Loop until sigma = 1 or for 200 matches
    matches = 0
    wins = 0
    while matches < 200 and agent_rating.sigma > 1:
        matches += 1
        # Get random opponent with mu in range(agent.mu - beta, agent.mu + beta)
        if agent_rating.sigma > 2:
            # we use initial_mu since agent.mu moves a lot during the first evaluations
            range_mu = initial_mu
        else:
            range_mu = agent_rating.mu

        opponent_listitem = get_opponent_in_range(ratings, range_mu - 2 * ts.beta, range_mu + 2 * ts.beta)
        opponent = PPO.load('policy/'+opponent_listitem["name"])
        op_rating = opponent_listitem["rating"]

        # Play a best of 9 match
        score_diff = 0
        agent_score = 0
        op_score = 0
        for i in range(9):
            # Play episode
            obs = env.reset()
            done = False

            while not done:
                # TODO: make this work with arbitrary agents. Not always the same agents one the same team
                surr_actions = []
                for i in range(team_size):
                    surr_actions.append(agent.predict(obs[i])[0])
                for i in range(team_size):
                    surr_actions.append(opponent.predict(obs[i+team_size])[0])

                obs, reward, done, info = env.step(np.asarray(surr_actions))

            result = info['result']
            score_diff += result

            if result > 0:
                agent_score += 1
            elif result < 0:
                op_score += 1

            if agent_score == 5 or op_score == 5:
                break

        # Update agent rating
        if score_diff > 0:
            wins += 1
            agent_rating, _ = rate_1vs1(agent_rating, op_rating)
        else:
            _, agent_rating = rate_1vs1(op_rating, agent_rating, drawn=(score_diff == 0))

        # TODO remove plotting
        print(agent_rating, score_diff, opponent_listitem['rating'].mu)
        mus.append(agent_rating.mu)

    # TODO remove plotting
    plt.plot(mus)
    plt.ylabel('mu')
    plt.show()

    # Put agent.mu and agent.sigma in redis
    print('Agent {} rating - matches: {} - wins: {} - mu: {} - sigma: {}'.format(agent_idx, matches, wins,
                                                                                 agent_rating.mu, agent_rating.sigma))
    ratings[agent_idx]["rating"] = agent_rating
env.close()
