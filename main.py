import gym
import json
import time
from env_wrapper import EnvWithExtraRandomStates
from genetic_algo import genetic_algorithm
from pi_approx import PiApproximationWithNN, PiApproximationWithFourier
from rl_algo import actor_critic, reinforce
from value_approx import ValueApproximationWithNN, ValueApproximationWithFourier


def objective(env, gamma, alpha, order, num_episodes, V_weights, V_c, pi_weights, pi_c,
              env_render=False) -> float:
    """ Evaluates the performance of the learning agent (policy module and value function)

        The function will be passed to genetic_algorithm and is used to return the scores
        (average rewards) of the current learning agent. It will create the policy module
        and value function module based on the parameters, and run the chosen RL algorithm

        Args:
            env: gym env
            gamma: discount factor
            alpha: learning rate (not used for genetic_algorithm)
            order: the order of the function approximation
            num_episodes: the number of episodes to run
            V_weights: a bitstring used to create value function with Fourier cosine
            V_c: a bistring used to create value function with Fourier cosine
            pi_weights: a bitstring used to create policy approximation with Fourier cosine
            pi_c: a bitstring used to create policy approximation with Fourier cosine
            env_render: whether to render the env or not

        Returns:
            The average rewards of the learning agent
    """
    pi = PiApproximationWithFourier(env.observation_space.shape[0], env.action_space.n, order=order,
                                    weight_values=pi_weights, c_values=pi_c)
    V = ValueApproximationWithFourier(env.observation_space.shape[0],
                                      order=order, weight_values=V_weights, c_values=V_c)
    score = reinforce(env, gamma, num_episodes, pi, V, env_render=env_render)
    return score


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    # env = EnvWithExtraRandomStates('CartPole-v0', extra_states=3)
    env = gym.make('CartPole-v0')
    rl_kwargs = {
        'alpha': 3e-4,
        'gamma': 1.,
        'num_episodes': 1,
        'order': 1,
        'env_render': False
    }
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    num_features = (rl_kwargs['order'] + 1) ** num_states
    V_bounds = [[-5, 5] for _ in range(num_features)]
    pi_bounds = [[-5, 5] for _ in range(num_actions * num_features)]

    genetic_kwargs = {
        'num_itrs': 200,
        'num_pops': 25,
        'n_bits_for_weights': 16,
        'n_bits_for_c': rl_kwargs['order'] + 1
    }

    records, best_individual = genetic_algorithm(objective, V_bounds, pi_bounds,
                                                 num_c=num_features * num_states, env=env, **genetic_kwargs,
                                                 **rl_kwargs)

    V = ValueApproximationWithFourier(num_states, order=rl_kwargs['order'], weight_values=best_individual[0],
                                      c_values=best_individual[1])
    pi = PiApproximationWithFourier(num_states, num_actions, order=rl_kwargs['order'], weight_values=best_individual[2],
                                    c_values=best_individual[3])
    print('---------------')
    print(env)
    print(V)
    print(pi)
    # pi = PiApproximationWithNN(num_states, num_actions, alpha=3e-4)
    # V = ValueApproximationWithNN(num_states, alpha=3e-4)
    # reinforce(env, rl_kwargs['gamma'], rl_kwargs['num_episodes'], pi, V, rl_kwargs['env_render'])


if __name__ == '__main__':
    main()
