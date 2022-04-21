from pi_approx import PiApproximation
from value_approx import ValueApproximation


def actor_critic(env, gamma: float, num_episodes: int,
                 pi: PiApproximation, V: ValueApproximation, env_render=False):
    total_rewards = 0
    for i in range(num_episodes):
        current_s, done = env.reset(), False
        I = 1
        while not done:
            if env_render:
                env.render()

            action, log_prob = pi(current_s)
            next_s, reward, done, _ = env.step(action)

            G = reward + gamma * V(next_s)
            delta = G - V(current_s)
            V.update(current_s, G)

            pi.update(log_prob, I, delta)
            I *= gamma
            current_s = next_s
            total_rewards += reward
        if i % 20 == 1:
            print('Finished %dth episode, total avg reward: %f' % (i, total_rewards / 20))
            total_rewards = 0
    return total_rewards / num_episodes


def generate_episode(env, pi, env_render):
    states = [env.reset()]
    actions = []
    rewards = [0]
    log_probs = []

    done = False
    current_s = states[0]
    while not done:
        if env_render:
            env.render()
        action, log_prob = pi(current_s)
        actions.append(action)
        log_probs.append(log_prob)

        next_s, reward, done, info = env.step(action)
        states.append(next_s)
        rewards.append(reward)
        current_s = next_s

    return states, actions, rewards, log_probs


def reinforce(
    env,  # open-ai environment
    gamma: float,
    num_episodes: int,
    pi: PiApproximation,
    V: ValueApproximation, env_render=False) -> [float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episode.
    """
    total_reward = 0
    for i in range(num_episodes):
        states, actions, rewards, log_probs = generate_episode(env, pi, env_render)
        T = len(rewards) - 1
        for t in range(0, T):
            G = sum(gamma ** (k - t - 1) * rewards[k] for k in range(t + 1, T))
            s, a = states[t], actions[t]
            delta = G - V(s)
            V.update(s, G)
            pi.update(log_probs[t], gamma ** t, delta)
        total_reward += sum(rewards)
        # if i % 20 == 0:
        #     print('Finished %dth episode, G_0=%.2f' % (i, G_values[0]))
    return total_reward / num_episodes
