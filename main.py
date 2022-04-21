import gym
from pi_approx import PiApproximationWithNN, PiApproximationWithFourier
from value_approx import ValueApproximationWithNN, ValueApproximationWithFourier
from rl_algo import actor_critic, reinforce
from genetic_algo import genetic_algorithm


def objective(env, gamma, alpha, order, num_episodes, V_weights, V_c, pi_weights, pi_c,
              env_render=False) -> float:
    """Creates a ValueApproximation based on values array, and run rl algo to get the score"""
    pi = PiApproximationWithFourier(env.observation_space.shape[0], env.action_space.n, alpha, order=order,
                                    weight_values=pi_weights, c_values=pi_c)
    V = ValueApproximationWithFourier(env.observation_space.shape[0], alpha,
                                      order=order, weight_values=V_weights, c_values=V_c)
    score = reinforce(env, gamma, num_episodes, pi, V, env_render=env_render)
    return score


def main():
    env = gym.make('CartPole-v0')
    rl_kwargs = {
        'alpha': 3e-4,
        'gamma': 1.,
        'num_episodes': 1,
        'order': 1,
    }
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    num_features = (rl_kwargs['order'] + 1) ** num_states
    V_bounds = [[-5, 5] for _ in range(num_features)]
    pi_bounds = [[-5, 5] for _ in range(num_actions * num_features)]
    n_iter = 1000
    genetic_algorithm(objective, V_bounds, pi_bounds, n_iter, num_pops=10, n_bits_for_weights=16,
                      n_bits_for_c=rl_kwargs['order'] + 1,
                      num_c=num_features * num_states, env=env, **rl_kwargs)
    # print('Done!')
    # print('Best score: %.2f' % score)
    # pi = PiApproximationWithNN(env.observation_space.shape[0], num_actions, rl_kwargs['alpha'])
    # V = ValueApproximationWithFourier(env.observation_space.shape[0], rl_kwargs['alpha'], rl_kwargs['order'])
    # actor_critic(env, rl_kwargs['gamma'], rl_kwargs['num_episodes'], pi, V, env_render=False)


if __name__ == '__main__':
    main()
