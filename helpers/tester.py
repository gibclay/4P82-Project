"""
This function tests an individual on a set of test data
returns an accuracy rating.
returns hits and accuracy
"""

from typing import Tuple


def tester(individual: any, test_data: any) -> Tuple[int, float]:
    length = len(test_data)
    hits = 0
    accuracy = 0

    for entry in test_data:
        guess = individual(*entry[:7])
        if guess >= 0.0 and entry[7] == 0:  # Cammeo
            hits += 1
        elif guess < 0.0 and entry[7] == 1:  # Osmancik
            hits += 1

    accuracy = hits * 100 / length
    return hits, accuracy

if __name__ == "__main__":
    pass