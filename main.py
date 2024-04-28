import operator
import math
import random
from functools import partial
import numpy
from deap import algorithms, base, creator, tools, gp
import os

from helpers.args_reader import *
from helpers.plotter import plotter
from helpers.protected_div import *
from helpers.evaluate import *
from helpers.rice_reader import rice_reader
from helpers.tester import tester

import matplotlib.pyplot as plt

# Based on the example code from
# https://deap.readthedocs.io/en/master/examples/gp_symbreg.html

# if not os.path.exists("stats"):
#   os.mkdir("stats")

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

# Add terminals to GP language
pset.addEphemeralConstant("rand1", partial(random.randint, -1, 1))

for i in range(5):
  pset.addEphemeralConstant(f"{chr(65+i)}", partial(random.uniform, -1, 1))

pset.renameArguments(
  ARG0="x1", ARG1="x2", ARG2="x3", ARG3="x4", ARG4="x5", ARG5="x6", ARG6="x7"
)

# Create Individuals
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Register evaluation function
toolbox.register("evaluate", evalSymbReg, data=TRAINING_DATA, toolbox=toolbox)

# Register genetic operators
toolbox.register("select", tools.selTournament, tournsize=T_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=5, max_=12)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=16))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=16))

def k_ring_swap(islands):
  amount = len(islands)

  for i in range(POP_SIZE // len(islands)):
    ind = random.randint(0, POP_SIZE-1)

    islands[i%amount][ind], islands[(i+1)%amount][ind] = islands[(i+1)%amount][ind], islands[i%amount][ind]

def safe_log(x):
  if x > 0:
    return math.log(x)
  return 0


stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
stats.register("avg", numpy.mean)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

NUM_ISLANDS = 6
islands = [toolbox.population(n=POP_SIZE) for _ in range(NUM_ISLANDS)]

RUNS = 10

averages = [[ 0.0 for _ in range(MAX_GENERATIONS) ] for _ in range(NUM_ISLANDS)]
bests    = [[ 0.0 for _ in range(MAX_GENERATIONS) ] for _ in range(NUM_ISLANDS)]

for run in range(RUNS):
  random.seed(run)

  for island_num, pop in enumerate(islands):
    # The evaluation function is applied to all members of the population.  
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    # The fitness is assigned to each individual.
    for ind, fit in zip(pop, fitnesses):
      ind.fitness.values = fit

    # Number of generations.
    for g in range(MAX_GENERATIONS):
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
      if g % 10 == 0:
        k_ring_swap(islands)
      
      # Record the averages.
      avg = 0
      for ind in pop:
        avg += ind.fitness.values[0]
      avg /= len(pop)
      averages[island_num][g] += avg

      # Record the bests.
      best = 0
      for ind in pop:
        fit = ind.fitness.values[0]
        if fit > best:
          best = fit
      bests[island_num][g] += best

for island_num in range(len(averages)):
  for g in range(len(averages[0])):
    averages[island_num][g] /= RUNS
    bests[island_num][g] /= RUNS

avgs_to_plot = []
bests_to_plot = []

for g in range(MAX_GENERATIONS):
  avg = 0
  best = 0
  
  for island_num in range(NUM_ISLANDS):
    avg += averages[island_num][g]
    best += bests[island_num][g]
  
  avg /= NUM_ISLANDS
  best /= NUM_ISLANDS

  avg = 100 - 100*avg/len(TRAINING_DATA)
  best = 100 - 100*best/len(TRAINING_DATA)

  avgs_to_plot.append(avg)
  bests_to_plot.append(best)


for i in range(NUM_ISLANDS):
  plt.clf()
  plt.plot(avgs_to_plot, color="green", label="Average (Mean)")
  plt.plot(bests_to_plot, color="red", label="Best (Mean)")
  plt.legend(loc="upper left")
  plt.xlabel("Generation")
  plt.ylabel(f'Fitness')
  plt.title(f'Average & Best Fitness for Individuals through All Islands')
  plt.savefig(f"plot.png")

# Print the best of each island.
with open("HOF.txt", "w") as file:
  for pop in islands:
    file.write(str(tools.selBest(pop, k=1)[0]) + "\n")
    fitness, accuracy = tester(toolbox.compile(expr=tools.selBest(pop, k=1)[0]), test_data=TESTING_DATA)
    print(
        "Best individual accuracy: "
        + str(accuracy)
        + "\nIt had "
        + str(fitness)
        + " hits out of "
        + str(len(TESTING_DATA))
        + " data points"
    )