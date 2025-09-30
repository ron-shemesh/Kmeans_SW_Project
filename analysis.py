import sys
import pandas as pd
from sklearn.metrics import silhouette_score
import kmeans
import symnmf_c as sf

#Import the required functions from symnmf.py, define constants for Kmeans
from symnmf import initialize_H, derive_clusters_from_H
MAX_ITER = 300
EPSILON = 0.0001



def parse_arguments():
    """
    Parses and validates cmd arguments for the analysis script.
    Returns the number of clusters and filename.
    """

    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
    
    try:
        k = int(sys.argv[1])
        filename = sys.argv[2]
        return k, filename
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)




def main():
    """
    The main routine of the analysis.
    it loads the data, runs both clustering algorithms seperately,
    and prints the silhouette scores.
    """
    k, filename = parse_arguments()
    
    try:
        data_points = pd.read_csv(filename, header=None).values.tolist()
    except FileNotFoundError:
        print("An Error Has Occurred")
        sys.exit(1)

    # SymNMF Analysis - Calculates W, initializes H, iterates to get the final H, derives clusters and calculates silhouette score
    W = sf.norm(data_points)
    H_initial = initialize_H(W, k)
    final_H = sf.SymNMF(H_initial, W)
    symnmf_labels = derive_clusters_from_H(final_H)
    score_symnmf = silhouette_score(data_points, symnmf_labels)

    # K-means Analysis - Runs the K-means algorithm, derives clusters, and calculates silhouette score
    centroids = kmeans.kmeans_routine(k, data_points, MAX_ITER, EPSILON)
    kmeans_labels = kmeans.assign_clusters(data_points, centroids)
    score_kmeans = silhouette_score(data_points, kmeans_labels)

    # Print Final Results
    print(f"nmf: {score_symnmf:.4f}")
    print(f"kmeans: {score_kmeans:.4f}")

if __name__ == "__main__":
    main()
