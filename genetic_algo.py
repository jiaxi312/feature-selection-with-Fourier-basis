import numpy as np
from random import choice
from numpy.random import rand, randint
from collections import namedtuple


def decode(bounds, n_bits, bitstring):
    # Reference: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    """Decodes the bitstring to a list of numbers"""
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


def bitlist_2_integers(n_bits, bitlist):
    values = []
    for i in range(0, len(bitlist), n_bits):
        bistring = ''.join((str(val) for val in bitlist[i:i + n_bits]))
        values.append(int(bistring, 2))
    return values


def evolve(parent1, parent2):
    # reference: https://www.geeksforgeeks.org/genetic-algorithms/
    child = []
    for i, j in zip(parent1, parent2):
        prob = np.random.rand()
        if prob < 0.45:
            child.append(i)
        elif prob < 0.9:
            child.append(j)
        else:
            child.append(randint(0, 2))
    return child


def modified_genetic_algorithm(fitness_func, V_bounds, pi_bounds, num_itrs, num_pops,
                               n_bits_for_weights, n_bits_for_c, num_c, **kwargs):
    """ Implements the modified genetic algorithm

    In classic GA, a fitness function is used to evaluate the performance for each individual.
    Individuals are sorted based on fitness, and two best individual will be selected to make an
    offspring. In the modified version, top 10% of individuals will go to next iteration directly.
    Then, top 50% of individuals will be selected random to generate the remaining 90% populations.

    Args:
        fitness_func: a function used to evaluate the goodness of each solution
        V_bounds: an array of tuple(2) specifies the range for each weights in Value approximation model
        pi_bounds: an array of tuple(2) specifies the range for each weights in Pi approximation model
        num_itrs: the number of iterations the GA would run
        num_pops: the population size
        n_bits_for_weights: number of bit for weight representation
        n_bits_for_c: number of bit for C representation
        num_c: the size of C vector
    """

    Individual = namedtuple('Individual', ['num_epoch', 'V_w_bitstring', 'V_c_bitstring',
                                           'pi_w_bistring', 'pi_c_bistring', 'fitness'])

    # Create the first generation with random values
    populations = [Individual(0,
                              randint(0, 2, size=n_bits_for_weights * len(V_bounds)),
                              randint(0, 2, size=n_bits_for_c * num_c),
                              randint(0, 2, size=n_bits_for_weights * len(pi_bounds)),
                              randint(0, 2, size=n_bits_for_c * num_c),
                              0)
                   for _ in range(num_pops)]

    records = []
    best_individual = None
    # Perform genetic algorithm
    for i in range(num_itrs):
        populations_with_fitness = []

        # evaluate each individual using the objective function
        for individual in populations:
            V_weights = decode(V_bounds, n_bits_for_weights, individual.V_w_bitstring)
            V_c = bitlist_2_integers(n_bits_for_c, individual.V_c_bitstring)
            pi_weights = decode(pi_bounds, n_bits_for_weights, individual.pi_w_bistring)
            pi_c = bitlist_2_integers(n_bits_for_c, individual.pi_c_bistring)
            populations_with_fitness.append(individual._replace(
                fitness=fitness_func(**kwargs, V_weights=V_weights, V_c=V_c, pi_weights=pi_weights, pi_c=pi_c)))

        populations = populations_with_fitness
        populations.sort(key=lambda x: x.fitness, reverse=True)

        new_generation = []

        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = int((10 * num_pops) / 100)
        new_generation.extend(populations[:s])

        # From 50% of fittest population, Individuals
        # will mate to produce offspring
        s = int((90 * num_pops) / 100)
        for _ in range(s):
            parent1 = choice(populations[:len(populations) // 2])
            parent2 = choice(populations[:len(populations) // 2])
            child = Individual(i + 1, evolve(parent1.V_w_bitstring, parent2.V_w_bitstring),
                               evolve(parent1.V_c_bitstring, parent2.V_c_bitstring),
                               evolve(parent1.pi_w_bistring, parent2.pi_w_bistring),
                               evolve(parent1.pi_c_bistring, parent2.pi_c_bistring),
                               0)
            new_generation.append(child)

        print("Generation:{}\tPopulation size: {}\t\tBest Fitness: {} "
              .format(i, len(populations), populations[0].fitness))

        records.append(populations[0].fitness)
        best_individual = populations[0]
        populations = new_generation

    V_weights = decode(V_bounds, n_bits_for_weights, best_individual.V_w_bitstring)
    V_c = bitlist_2_integers(n_bits_for_c, best_individual.V_c_bitstring)
    pi_weights = decode(pi_bounds, n_bits_for_weights, best_individual.pi_w_bistring)
    pi_c = bitlist_2_integers(n_bits_for_c, best_individual.pi_c_bistring)

    return records, [V_weights, V_c, pi_weights, pi_c]


def classic_genetic_algorithm(fitness_func, V_bounds, pi_bounds, num_itrs, num_pops,
                              n_bits_for_weights, n_bits_for_c, num_c, **kwargs):
    """ Implements the classic genetic algorithm.

        In classic GA, a fitness function is used to evaluate the performance for each individual.
        Individuals are sorted based on fitness, and two best individual will be selected to make an
        offspring.

        Args:
            fitness_func: a function used to evaluate the goodness of each solution
            V_bounds: an array of tuple(2) specifies the range for each weights in Value approximation model
            pi_bounds: an array of tuple(2) specifies the range for each weights in Pi approximation model
            num_itrs: the number of iterations the GA would run
            num_pops: the population size
            n_bits_for_weights: number of bit for weight representation
            n_bits_for_c: number of bit for C representation
            num_c: the size of C vector
    """
    Individual = namedtuple('Individual', ['num_epoch', 'V_w_bitstring', 'V_c_bitstring',
                                           'pi_w_bistring', 'pi_c_bistring', 'fitness'])

    # Create the first generation with random values
    populations = [Individual(0,
                              randint(0, 2, size=n_bits_for_weights * len(V_bounds)),
                              randint(0, 2, size=n_bits_for_c * num_c),
                              randint(0, 2, size=n_bits_for_weights * len(pi_bounds)),
                              randint(0, 2, size=n_bits_for_c * num_c),
                              0)
                   for _ in range(num_pops)]

    records = []
    best_individual = None

    # Perform genetic algorithm
    for i in range(num_itrs):
        populations_with_fitness = []
        # evaluate each individual using the objective function
        for individual in populations:
            V_weights = decode(V_bounds, n_bits_for_weights, individual.V_w_bitstring)
            V_c = bitlist_2_integers(n_bits_for_c, individual.V_c_bitstring)
            pi_weights = decode(pi_bounds, n_bits_for_weights, individual.pi_w_bistring)
            pi_c = bitlist_2_integers(n_bits_for_c, individual.pi_c_bistring)
            populations_with_fitness.append(individual._replace(
                fitness=fitness_func(**kwargs, V_weights=V_weights, V_c=V_c, pi_weights=pi_weights, pi_c=pi_c)))

        populations = populations_with_fitness
        populations.sort(key=lambda x: x.fitness, reverse=True)
        populations = populations[:num_pops]

        parent1 = populations[0]
        parent2 = populations[2]
        child = Individual(i + 1, evolve(parent1.V_w_bitstring, parent2.V_w_bitstring),
                           evolve(parent1.V_c_bitstring, parent2.V_c_bitstring),
                           evolve(parent1.pi_w_bistring, parent2.pi_w_bistring),
                           evolve(parent1.pi_c_bistring, parent2.pi_c_bistring),
                           0)
        populations.append(child)

        print("Generation:{}\tPopulation size: {}\t\tBest Fitness: {} "
              .format(i, len(populations), populations[0].fitness))

        records.append(populations[0].fitness)
        best_individual = populations[0]

    V_weights = decode(V_bounds, n_bits_for_weights, best_individual.V_w_bitstring)
    V_c = bitlist_2_integers(n_bits_for_c, best_individual.V_c_bitstring)
    pi_weights = decode(pi_bounds, n_bits_for_weights, best_individual.pi_w_bistring)
    pi_c = bitlist_2_integers(n_bits_for_c, best_individual.pi_c_bistring)

    return records, [V_weights, V_c, pi_weights, pi_c]
