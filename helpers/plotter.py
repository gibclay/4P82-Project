from matplotlib import pyplot as plt


def plotter(bestOver10, avgOver10):
    plt.plot(bestOver10, color="red")
    plt.plot(avgOver10, color="green")
    plt.xlabel("Generations")
    plt.ylabel("Average Inaccuracy (%)")
    plt.title("Best and Average Fitness over Generations (across 10 Seeds)")
    plt.savefig("plot.png")
