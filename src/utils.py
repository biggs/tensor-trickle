""" utils.py: Basic utilities."""
import numpy as np

def np_str(arr, max_lines=10):
    """ Convert an array to a string in a logging-friendly way."""
    string = np.array_str(
        arr, precision=3, suppress_small=False, max_line_width=120)
    strings = string.split("\n")[:max_lines]
    continuation = " "*int(len(strings[0])/2 - 3) + ". . ."
    if not len(strings) < max_lines:
        strings += [continuation]
    return "\n".join([str(arr.shape)] + strings)
