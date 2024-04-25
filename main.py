import operator
import math
import random
from functools import partial
import numpy
from deap import algorithms, base, creator, tools, gp
from scoop import futures

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

def setup():
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
  creator.create("FitnessMin", base.Fitness, weights=(1.0,))
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

  toolbox.register("map", futures.map)
  # best 15 are moved to the adjacent island.
  toolbox.register("migrate", tools.migRing, k=15, selection=tools.selBest)

  

def main():
  setup()
  training_length = int(len(data) * TRAINING_SET_SIZE)
  print("Training set data size: " + str(training_length))

  pop = toolbox.population(n=POP_SIZE)
  hof = tools.HallOfFame(HOF_SIZE)

  stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
  stats_size = tools.Statistics(len)
  mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
  mstats.register("avg", numpy.mean)
  mstats.register("std", numpy.std)
  mstats.register("min", numpy.min)
  mstats.register("max", numpy.max)

  # FREQ is how long they evolve independently.
  NGEN, FREQ = MAX_GENERATIONS, 10

  toolbox.register("algorithm", algorithms.eaSimple, toolbox=toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=FREQ, verbose=False)
  islands = [toolbox.population(n=POP_SIZE) for i in range(5)]
  
  for i in range(0, NGEN, FREQ):
    results = toolbox.map(toolbox.algorithm, islands)
    islands = [(island, logbook) for island, logbook in results]
    toolbox.migrate(islands)

  # bestVals, avgVals = logbook.chapters["fitness"].select("max", "avg")
  return islands

if __name__ == "__main__":
  islands = main()
