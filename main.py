import gym
from pi_approx import PiApproximationWithNN, PiApproximationWithFourier
from value_approx import ValueApproximationWithNN, ValueApproximationWithFourier
from rl_algo import actor_critic, reinforce
from genetic_algo import genetic_algorithm
from env_wrapper import EnvWithExtraRandomStates

num = 1


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
    pi = PiApproximationWithFourier(env.observation_space.shape[0], env.action_space.n, alpha, order=order,
                                    weight_values=pi_weights, c_values=pi_c)
    V = ValueApproximationWithFourier(env.observation_space.shape[0], alpha,
                                      order=order, weight_values=V_weights, c_values=V_c)
    score = actor_critic(env, gamma, num_episodes, pi, V, env_render=env_render)
    global num
    num += 1
    return score


def main():
    env = EnvWithExtraRandomStates('CartPole-v0')
    # env = gym.make('CartPole-v0')
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
    n_iter = 10
    genetic_algorithm(objective, V_bounds, pi_bounds, n_iter, num_pops=25, n_bits_for_weights=16,
                      n_bits_for_c=rl_kwargs['order'] + 1,
                      num_c=num_features * num_states, env=env, **rl_kwargs)
    print(f'Total number of times rl algo being called: {num}')
    # print('Done!')
    # print('Best score: %.2f' % score)
    # pi = PiApproximationWithNN(env.observation_space.shape[0], num_actions, rl_kwargs['alpha'])
    # V = ValueApproximationWithFourier(env.observation_space.shape[0], rl_kwargs['alpha'], rl_kwargs['order'])
    # actor_critic(env, rl_kwargs['gamma'], rl_kwargs['num_episodes'], pi, V, env_render=rl_kwargs['env_render'])


if __name__ == '__main__':
    main()
