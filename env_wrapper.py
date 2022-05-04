import gym
import numpy as np


class EnvWithExtraRandomStates:
    """ A wrapper class for the gym environment. The goal of this class is to generate some random numbers whenever
        the environment returns state array. For now, it would only append the random number to the end of the state
        array, and the number would stay the same.

        TODO: write functions to insert the random numbers to random positions in the array
    """

    def __init__(self, env_name, extra_states=2, random_states=False):
        self.env = gym.make(env_name)
        self.extra_states = extra_states
        self.original_states = self.env.observation_space.shape[0]
        self.env.observation_space.shape = np.array([self.original_states + self.extra_states])
        self.random_states = random_states

        self.constant_extra_states = self.create_random_arr(self.extra_states, low=-4, high=4)

    def render(self):
        self.env.render()

    def reset(self):
        states = self.env.reset()
        new_states = self._append_extra_states(states)
        return new_states

    def step(self, action):
        next_s, reward, done, info = self.env.step(action)
        new_states = self._append_extra_states(next_s)
        return new_states, reward, done, info

    def _append_extra_states(self, states):
        if not self.random_states:
            return np.append(states, self.constant_extra_states)

    @staticmethod
    def create_random_arr(n, low=0, high=1):
        rand_arr = low + (high - low) * np.random.rand(n)
        return rand_arr

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


if __name__ == '__main__':
    env = EnvWithExtraRandomStates('CartPole-v0')
    print(env.reset())
    print(EnvWithExtraRandomStates.create_random_arr(10, -3, 3))
