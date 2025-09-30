import sys
import pandas as pd
import numpy as np
import symnmf_c as sf

np.random.seed(1234) 


def initialize_H(W, k):
    """Initializes H matrix with uniform random values from the interval [0, 2 * sqrt(m/k)],
    where m is the average of the elements in W.
    Takes W (list of list), k (int) as input.
    Returns H (list of list)."""
    m = sf.avgMat(W)
    num_rows = len(W)
    upper_bound = 2 * np.sqrt(m / k)
    H = [[np.random.uniform(0, upper_bound) for _ in range(k)] for _ in range(num_rows)]
    return H


def derive_clusters_from_H(H_matrix):
    """Derives hard cluster assignment from the SymNMF result matrix H.
    Each data point is assigned to the cluster corresponding to the index
    of the maximum value in its row in H.
    Takes H_matrix (list of list) as input.
    Returns a list of cluster assignments (list of int)."""
    return np.argmax(H_matrix, axis=1).tolist()


def run_goal(k, goal, points):
    """Executes the relevant routine based on the input goal.
    Takes k (int), goal (str), points (list of list) as input.
    Returns the resulting matrix (list of list)."""

    goal_to_function_map = {
        "sym": sf.sym,
        "ddg": sf.ddg,
        "norm": sf.norm
    }

    if goal == "symnmf":
        # Full SymNMF process: calculate W, initialize H, and run algorithm
        W = sf.norm(points)
        H_initial = initialize_H(W, k)
        return sf.SymNMF(H_initial, W)
    elif goal in goal_to_function_map:
        return goal_to_function_map[goal](points)
    else:
        print("An Error Has Occurred")
        sys.exit(1)


def print_mat(matrix):
    """Prints the matrix in the required format."""
    for row in matrix:
        formatted_row = ','.join([f'{val:.4f}' for val in row])
        print(formatted_row)


def main():
    """Main function: parses arguments, loads data, executes routine, prints result."""
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]

        points = pd.read_csv(filename, header=None).values.tolist()
        result_matrix = run_goal(k, goal, points)
        print_mat(result_matrix)

    except (ValueError, FileNotFoundError):
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
