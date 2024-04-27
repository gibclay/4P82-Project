import operator
import math
import random
from functools import partial
import numpy
from deap import algorithms, base, creator, tools, gp

from helpers.args_reader import *
from helpers.plotter import plotter
from helpers.protected_div import *
from helpers.evaluate import *
from helpers.rice_reader import rice_reader
from helpers.tester import tester

# Based on the example code from
# https://deap.readthedocs.io/en/master/examples/gp_symbreg.html

random.seed(SEED)

toolbox = base.Toolbox()
data = rice_reader("data/delicious_rice.txt")
TRAINING_DATA, TESTING_DATA = (
  data[: math.floor(len(data) * TRAINING_SET_SIZE)],
  data[math.floor(len(data) * TRAINING_SET_SIZE) :],
)

# Add functions to GP language
pset = gp.PrimitiveSet("MAIN", 7)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_division, 2)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(min, 2)
# pset.addPrimitive(max, 2)

# Add terminals to GP language
# pset.addEphemeralConstant("rand1", partial(random.randint, -1, 1))
# pset.addEphemeralConstant("rand2", partial(random.randint, -1, 1))
# pset.addEphemeralConstant("rand3", partial(random.randint, -1, 1))

for i in range(5):
  pset.addEphemeralConstant(f"{chr(65+i)}", partial(random.uniform, -1, 1))

pset.renameArguments(
  ARG0="x1", ARG1="x2", ARG2="x3", ARG3="x4", ARG4="x5", ARG5="x6", ARG6="x7"
)

# Create Individuals
creator.create(
  "FitnessMin", base.Fitness, weights=(1.0,)
)  # positive weights -> maximization
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Register evaluation function
toolbox.register("evaluate", evalSymbReg, data=TRAINING_DATA, toolbox=toolbox)

# Register genetic operators
toolbox.register("select", tools.selTournament, tournsize=T_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=17)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate(
  "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox.decorate(
  "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)


if __name__ == "__main__":
  stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
  stats.register("avg", numpy.mean)
  stats.register("min", numpy.min)
  stats.register("max", numpy.max)

  NUM_ISLANDS = 12
  islands = [toolbox.population(n=POP_SIZE) for _ in range(NUM_ISLANDS)]

  # Each iteration deals with one island at a time.
  for pop in islands:
    # The evaluation function is applied to all members of the population.  
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    # The fitness is assigned to each individual.
    for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit

    # Number of generations.
    for g in range(100):
      # k individuals are selected from the population.
      pop = toolbox.select(pop, k=len(pop))

      # The existing population is duplicated.
      offspring = [toolbox.clone(ind) for ind in pop]

      # Crossover is applied to "a portion of consecutive individuals."
      for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
          toolbox.mate(child1, child2)
          # Their existing fitnesses (from being cloned) are deleted.
          del child1.fitness.values, child2.fitness.values

      # A percentage of the newly crossed over offspring is mutated.
      for mutant in offspring:
        if random.random() < P_MUTATION:
          toolbox.mutate(mutant)
          # Make sure to delete the existing fitness value.
          del mutant.fitness.values

      # Freshly produced individuals are filtered by the validity of their fitness.
      invalids = [ind for ind in pop if not ind.fitness.valid]
      # The remaining are given new fitness values.
      fitnesses = toolbox.map(toolbox.evaluate, invalids)
      for ind, fit in zip(invalids, fitnesses):
        ind.fitness.values = fit

      # If the generation count is a multiple of 5: 
      # Two random islands are selected and a couple of their individuals are traded between each other.
      if g % 5 == 0:
        island_one = random.randint(0, NUM_ISLANDS-1)
        island_two = random.randint(0, NUM_ISLANDS-1)

        for num_of_inds_swapped in range(POP_SIZE//NUM_ISLANDS):
          ind = random.randint(0, POP_SIZE-1)

          islands[island_one][ind], islands[island_two][ind] = islands[island_two][ind], islands[island_one][ind]

      # For this generation for this island, we write down the stats.
      record = stats.compile(pop)
      logbook = tools.Logbook()
      logbook.record(gen=0, evals=30, **record)

      gen, avg = logbook.select("gen", "avg")
      print(gen, avg)

  # Print the best of each island.
  for pop in islands:
    print(tools.selBest(pop, k=1)[0])