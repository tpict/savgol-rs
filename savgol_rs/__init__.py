import argparse
import json
from scipy.signal import savgol_filter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs="*", type=float, required=True)
parser.add_argument("-w", "--window", type=int, required=True)
parser.add_argument("-p", "--poly-order", type=int, required=True)
parser.add_argument("-d", "--derivative", type=int, required=True)

def run():
    args = parser.parse_args()
    try:
        result = savgol_filter(np.array(args.input), args.window, args.poly_order, args.derivative)
        print(json.dumps(list(result)))
    except ValueError as e:
        print(str(e))
