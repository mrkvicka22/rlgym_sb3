import os

import rlgym
import pickle
import numpy as np
from time import sleep
from trueskill import Rating, rate_1vs1, global_env, setup
from rlgym.utils.state_setters.default_state import DefaultState
from rlgym.utils.state_setters.state_setter import StateSetter
from stable_baselines3.ppo import PPO
from rlgym.utils.obs_builders import AdvancedObs, ObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, \
    NoTouchTimeoutCondition
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym.utils.action_parsers.action_parser import ActionParser
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

setup(draw_probability=0.01)
ts = global_env()


def save_ratings(ratings_database):
    with open("policy_ratings", "wb") as f:
        pickle.dump(ratings_database, f)


def get_policies(order_func, path):
    return list(sorted(os.listdir(path), key=order_func))


def get_opponent_in_range(ratings_database: dict, min_mu, max_mu):
    # Get MUs
    all_mus = np.asarray(
        [ratings_database['opponents'][i]["rating"].mu for i in range(len(ratings_database['opponents']))])

    # Get opponents in rating range
    op_indexes = np.where((all_mus >= min_mu) & (all_mus <= max_mu))[0]

    # If op_indexes.length == 0, get the closest opponent
    if len(op_indexes) == 0:
        # Get the closest opponent
        print("No opponents in mmr range, getting closest agent")
        mus_distance = np.abs(all_mus - (min_mu + max_mu) / 2)
        index = int(np.argmin(mus_distance))
    else:
        # Choose one
        index = int(np.random.choice(op_indexes))
    return ratings_database['opponents'][index]


# Initialize rating list if necessary (length == 0)
def initialize_ratings(order_func, path, reset_ratings: bool = False):
    try:
        with open("policy_ratings", "rb") as f:
            ratings_database = pickle.load(f)
        for model in get_policies(order_func, path):
            if model not in [model_item["name"] for model_item in ratings_database['agents']] and model not in [model_item["name"] for model_item in ratings_database['opponents']]:
                ratings_database['agents'].append({"name": model, "rating": Rating()})
    except FileNotFoundError:
        ratings_database = {'agents': [], 'opponents': []}
        policies = get_policies(order_func, path)
        for model in policies[1:]:
            ratings_database['agents'].append({"name": model, "rating": Rating()})
        ratings_database['opponents'].append({"name": policies[0], "rating": Rating(sigma=1e-4)})
    if reset_ratings:
        ratings_database = {'agents': [], 'opponents': []}
        policies = get_policies(order_func, path)
        for model in policies[1:]:
            ratings_database['agents'].append({"name": model, "rating": Rating()})
        ratings_database['opponents'].append({"name": policies[0], "rating": Rating(sigma=1e-4)})
    return ratings_database


def main(model_dir: str,
         order_func,
         obs: ObsBuilder = AdvancedObs(),
         state_setter: StateSetter = DefaultState(),
         action_parser: ActionParser = DiscreteAction(),
         max_number_matches_to_play: int = 200,
         best_of_how_many: int = 9,
         sigma_threshold: int = 1,
         team_size: int = 1,
         max_steps: int = 60 * 15 * 2,
         no_touch_steps: int = 500, ):
    env = rlgym.make(team_size=team_size, self_play=True, use_injector=True,
                     obs_builder=obs,
                     state_setter=state_setter,
                     terminal_conditions=[TimeoutCondition(max_steps), GoalScoredCondition(),
                                          NoTouchTimeoutCondition(no_touch_steps)],
                     action_parser=action_parser,
                     )

    while True:
        # Get reference Mu -> Returns last MU and last index
        ratings = initialize_ratings(order_func, model_dir)
        initial_mu = ratings['opponents'][-1]["rating"].mu
        agent_rating = Rating(mu=initial_mu)

        if len(ratings['agents']) == 0:
            print("No new model to evaluate, sleeping 10 minutes")
            sleep(600)
            continue
        agent = PPO.load(os.path.join(model_dir, ratings['agents'][0]['name']))

        # Loop until sigma = 1 or for 200 matches
        matches = 0
        wins = 0
        while matches < max_number_matches_to_play and agent_rating.sigma > sigma_threshold:
            matches += 1
            # Get random opponent with mu in range(agent.mu - beta, agent.mu + beta)
            if agent_rating.sigma > 2:
                # we use initial_mu since agent.mu moves a lot during the first evaluations
                range_mu = initial_mu
            else:
                range_mu = agent_rating.mu

            opponent_listitem = get_opponent_in_range(ratings, range_mu - 2 * ts.beta, range_mu + 2 * ts.beta)
            opponent = PPO.load(os.path.join(model_dir, opponent_listitem["name"]))
            op_rating = opponent_listitem["rating"]

            # Play a best of 9 match
            score_diff = 0
            agent_score = 0
            op_score = 0
            for i in range(best_of_how_many):
                # Play episode
                obs = env.reset()
                done = False

                while not done:
                    # TODO: make this work with arbitrary agents. Not always the same agents on the same team
                    surr_actions = np.concatenate(
                        [agent.predict(obs[:team_size])[0], opponent.predict(obs[team_size:])[0]])

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

            print(agent_rating, score_diff, opponent_listitem['rating'].mu)

        # Put agent.mu and agent.sigma in redis
        print('Agent {} rating - matches: {} - wins: {} - mu: {} - sigma: {}'.format(ratings['agents'][0]["name"],
                                                                                     matches,
                                                                                     wins,
                                                                                     agent_rating.mu,
                                                                                     agent_rating.sigma))
        ratings['agents'][0]["rating"] = agent_rating
        ratings['opponents'].append(ratings['agents'][0])
        ratings['agents'].pop(0)
        save_ratings(ratings)
    env.close()


if __name__ == '__main__':
    model_directoryss = 'testing'  # choose the directory in which you store your policies
    max_matches_to_play = 200  # how many matches do you want the evaluator to you use in order to determine the skill
    best_of = 9  # how many mini-matches (first to score a goal or reach timeout) do you want to have in each match
    sigma_th = 1  # the threshold at which the evaluation of agent is stopped as the mmr does not move enough
    order_function = lambda x: int(x.split("_")[2])  # this is a function that orders your models based on version.
    main(model_directoryss, order_function, action_parser=KBMAction(), sigma_threshold=2)
