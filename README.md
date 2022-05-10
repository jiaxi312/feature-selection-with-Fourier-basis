# Instruction
To run the program, simply run the code in `main.py`.

Package used in this project: `gym==0.12.1` and `torch==1.2.0`.

`test_genetic_algo()` in `main.py` is the function to run genetic algorithm with
Fourier basis function approximations. `rl_kwargs` in `test_genetic_algo` stores
the parameters of reinforcement learning algorithm, like learning rate, number of
episode, etc. No need to change those parameters, as learning rate is useless
in genetic algo and the number episode for RL algo is always 1 so that no gradient
descent update will happen. To see the rendered env, change the `env_render` in `rl_kwargs`
to `True`.

`genetic_kwargs` in `test_genetic_algo` stores the parameters of genetic algorithm.
`num_itrs` and `num_pop` control the number of iteration of GA and the
population size. `n_bits_for_weights` and `n_bits_for_c` control how many
bits used to represent weights and c.

# Brief Overview
There are two reinforcement learning algorithm (REINFORCE and actor-critic) implemented in `rl_algo.py`.'

The policy approximation modules are implemented in `pi_approx.py`. There are two modules, one using neural network 
and one using Fourier basis function approximation.

The value approximation modules are implemented in `value_approx.py`. There are two modules, one using neural network
and one using Fourier basis function approximation.

The genetic algorithm is implemented in `genetic_algo.py`. 
