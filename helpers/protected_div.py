"""
This function avoids divide by zero errors
"""


def safe_division(l, r):
    try:
        return l / r
    except ZeroDivisionError:
        return 1
