import math


def evalSymbReg(individual, data, toolbox):
    func = toolbox.compile(expr=individual)
    hits = 0

    for datum in data:
        # guess = func(datum[0], datum[1], datum[2], datum[3], datum[4], datum[5], datum[6])
        guess = func(*datum[:7])

        if guess >= 0.0 and datum[7] == 0:  # Cammeo
            hits += 1
        elif guess < 0.0 and datum[7] == 1:  # Osmancik
            hits += 1

    return (hits,)
