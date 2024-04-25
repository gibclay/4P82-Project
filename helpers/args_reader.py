"""
This file reads arguments from the ARGS file in the root folder as initialization parameters
"""

with open("ARGS", "r") as file:
    file_list = [f for f in file.readlines() if not f.startswith("#")]
    file_list = [f.strip() for f in file_list]

    POP_SIZE = int(file_list[0])
    P_CROSSOVER = float(file_list[1])
    P_MUTATION = float(file_list[2])
    MAX_GENERATIONS = int(file_list[3])
    HOF_SIZE = int(file_list[4])
    T_SIZE = int(file_list[5])
    DEPTH = int(file_list[6])
    SEED = int(file_list[7])
    TRAINING_SET_SIZE = float(file_list[8])
