from collections import Counter
import math

class KNN():
    def knn(data, query, k, dist_fn, choice_fn):
        neighbor_dist_and_indices = []
        for i, example in enumerate(data):
            distance = dist_fn(example[:-1], query)
            neighbor_dist_and_indices.append((distance, i))

        print('dist&indices : ',neighbor_dist_and_indices)
        sorted_neighbor_dist_and_indices = sorted(neighbor_dist_and_indices)

        k_nearest_neighbor_dist_and_indices = sorted_neighbor_dist_and_indices[:k]
        print('sorted k = 3 : ', k_nearest_neighbor_dist_and_indices)
        k_entries_labels = [data[i][1] for distance, i in k_nearest_neighbor_dist_and_indices]
        print('k_entries_labels',k_entries_labels)

        return k_nearest_neighbor_dist_and_indices, choice_fn(k_entries_labels)


    # mean is selected as choice_fn , if regression
    def mean(labels):
        return sum(labels)/len(labels)


    # mode is selected as choice_fn , if classification
    def mode(labels):
        return Counter(labels).most_common(1)[0][0]


    def euclidean_dist(p1, p2):
        sum_sq_dist = 0
        for i in range(len(p1)):
            sum_sq_dist += math.pow(p1[i] - p2[i], 2)
        return math.sqrt(sum_sq_dist)
