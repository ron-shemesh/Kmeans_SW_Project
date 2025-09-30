# HW1 - Clustering K-means in PY - RON SHEMESH - ID 212204200
import sys

# Note:  All arguments validation throughout the code are left from HW1

def read_txt(): # Note: this function is not used in analysis.py
    try:
        k_float = float(sys.argv[1])
        if not k_float.is_integer() or k_float < 1:
            raise ValueError
        k = int(k_float)
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)

    try:
        if len(sys.argv) > 2:
            iter_float = float(sys.argv[2])
            if not iter_float.is_integer() or iter_float < 1 or iter_float > 1000:
                raise ValueError
            iter = int(iter_float)
        else:
            iter = 400
    except ValueError:
        print("Incorrect maximum iteration!")
        sys.exit(1)

    vectors = []
    
    for line in sys.stdin:
        tokens = line.strip().replace(',', ' ').split()
        if not tokens:
            continue
        vector = [float(token) for token in tokens]
        vectors.append(vector)

    if not vectors:
        print("An Error Has Occurred")
        sys.exit(1)

    return vectors, k, iter

def initialize_centroids(vectors, k): # Picks the first k vectors as centroids, it used indirectly in analysis.py, via kmeans_routine
    if k > len(vectors):
        print("Incorrect number of clusters!") #k cannot be larger than the number of data points
        sys.exit(1)

    return [vec[:] for vec in vectors[:k]]

def assign_clusters(vectors, centroids): # Assigns each vector to the closest centroid, it is used directly in analysis.py
    cent_groups = []
    for vec in vectors:
        min_distance = float('inf')
        closest_index = -1
        for i, centroid in enumerate(centroids):
            distance = calc_distance(vec, centroid)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        if closest_index == -1:
            print("An Error Has Occurred")
            sys.exit(1)
        cent_groups.append(closest_index)
    return cent_groups


def update_centroids(vectors, cent_groups,k): # Updates centroids based on the assigned vectors, it is used indirectly in analysis.py, via kmeans_routine
    new_centroids = [[0.0] * len(vectors[0]) for j in range(k)]
    count = [0] * k

    for i, vec in enumerate(vectors):
        group_index = cent_groups[i]
        if group_index >= k:
            print("Incorrect number of clusters!")
            sys.exit(1)
        count[group_index] += 1
        for j in range(len(vectors[0])):
            new_centroids[group_index][j] += vec[j]
    
    for i in range(k):
        if count[i]  == 0:
            continue
        for j in range(len(vectors[0])):
            new_centroids[i][j] /= count[i]
    
    return new_centroids
    

def calc_distance(vec1, vec2): # Helper function to calculate the euclidean distance between two vectors,
        res = 0.0
        for i in range(len(vec1)):
            diff = float(vec1[i]) - float(vec2[i])
            res += diff * diff
        return res ** 0.5

def kmeans_routine(k, vectors, iter, eps): # This is the main routine of the Kmeans algorithm, it is used directly in analysis.py
    centroids = initialize_centroids(vectors, k)

    for i in range(iter):
        cent_groups = assign_clusters(vectors, centroids)
        new_centroids = update_centroids(vectors, cent_groups,k)

        # Check for convergence
        max_shift = 0.0
        if new_centroids and centroids: # Ensure lists are not empty
            max_shift = max(calc_distance(centroids[j], new_centroids[j]) for j in range(k))
        
        centroids = new_centroids

        if max_shift < eps:
            break

    return centroids 


def main(): # Note: main is not used in analysis.py
    vectors, k, iter = read_txt()
    
    final_centroids = kmeans_routine(k, vectors, iter, 0.001)

    for cent in final_centroids:
        print(",".join(f"{val:.4f}" for val in cent))
        
if __name__ == "__main__":
    main()