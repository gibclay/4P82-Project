import random
from typing import List

"""
this function takes the path to a data file and reads returns the data as a list of lists.
"""


def rice_reader(filepath: str) -> List[list]:
    with open(filepath, "r") as file:
        # remove the commented out lines in the datafile
        data = [
            f for f in file.readlines() if not (f.startswith("%") or f.startswith("@"))
        ]
        data = [f.strip() for f in data]
        # This next line is a bit crazy. So basically:
        # data is currently a string in the form of '<float>,...,<float>,<integer>'.
        # and this is taken to produce a string of floats followed by an integer
        data = [
            [float(v) for v in values[:-1]] + [int(values[-1])]
            for values in [string.split(",") for string in data]
        ]
        data = shake_rice(data)
    return data


"""
This function scrambles the data input to it.
"""


def shake_rice(data: List[list]) -> List[list]:
    random.shuffle(data)
    return data

if __name__ == "__main__":
    print(len(rice_reader("../data/delicious_rice.txt")))