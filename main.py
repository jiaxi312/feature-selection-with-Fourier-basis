import json
from env_wrapper import EnvWithExtraRandomStates
from genetic_algo import classic_genetic_algorithm, modified_genetic_algorithm
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
    score = actor_critic(env, gamma, num_episodes, pi, V, env_render=env_render)
    return score


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def test_gradient_descent():
    """Test gradient descent method with neural network"""
    for i in [0]:
        print('---------------------')
        env = EnvWithExtraRandomStates('MountainCar-v0', extra_states=i)
        rl_kwargs = {
            'alpha': 3e-4,
            'gamma': 1.,
            'num_episodes': 1000,
            'env_render': True
        }
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        pi = PiApproximationWithNN(num_states, num_actions, rl_kwargs['alpha'])
        V = ValueApproximationWithNN(num_states, rl_kwargs['alpha'])
        _, records = reinforce(env, pi=pi, V=V, gamma=rl_kwargs['gamma'],
                               num_episodes=rl_kwargs['num_episodes'], env_render=rl_kwargs['env_render'])


def test_genetic_algo():
    """Test genetic algorithm with Fourier Basis approximation """
    for i in [0]:
        print('----------------------------')
        env = EnvWithExtraRandomStates('CartPole-v0', extra_states=i)
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
            'num_itrs': 150,
            'num_pops': 50,
            'n_bits_for_weights': 16,
            'n_bits_for_c': rl_kwargs['order'] + 1
        }

        records, best_individual = modified_genetic_algorithm(objective, V_bounds, pi_bounds,
                                                              num_c=num_features * num_states, env=env,
                                                              **genetic_kwargs,
                                                              **rl_kwargs)
        V = ValueApproximationWithFourier(num_states, order=rl_kwargs['order'], weight_values=best_individual[0],
                                          c_values=best_individual[1])
        pi = PiApproximationWithFourier(num_states, num_actions, order=rl_kwargs['order'],
                                        weight_values=best_individual[2],
                                        c_values=best_individual[3])
        print('---------------')
        print(env)
        print(V)
        print(pi)


def main():
    test_genetic_algo()


if __name__ == '__main__':
    main()
