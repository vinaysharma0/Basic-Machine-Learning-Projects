from collections import Counter
import math

def knn(data, query, k, dist_fn, choice_fn):
	neighbor_dist_and_indices = []
	for i, example in enumerate(data):
		distance = dist_fn(example[:-1], query)

		neighbor_dist_and_indices.append((distance, i))

	sorted_neighbor_dist_and_indices = sorted(neighbor_dist_and_indices)

	k_nearest_neighbor_dist_and_indices = sorted_neighbor_dist_and_indices[:k]

	k_entries_labels = [data[i][1] for distance, i in k_nearest_neighbor_dist_and_indices]

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


def main():
	'''
    # Regression Data
    # 
    # Column 0: height (inches)
    # Column 1: weight (pounds)
    '''

	reg_data = [
        [65.75, 112.99],
        [71.52, 136.49],
        [69.40, 153.03],
        [68.22, 142.34],
        [67.79, 144.30],
        [68.70, 123.30],
        [69.80, 141.49],
        [70.01, 136.46],
        [67.90, 112.37],
        [66.49, 127.45],
    ]

	reg_query = [60]

	reg_k_nearest_neighbors, reg_pred = knn(reg_data, reg_query, k =3 , dist_fn = euclidean_dist, choice_fn = mean)

	print('reg_k_nearest_neighbors : {}, reg_pred : {}'.format(reg_k_nearest_neighbors, reg_pred))

	clf_data = [
	    [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0],
	]

	clf_query = [33]

	clf_k_nearest_neighbors, clf_pred = knn(clf_data, clf_query, k=3, distance_fn=euclidean_dist, choice_fn=mode)

	print('clf_k_nearest_neighbors : {}, clf_pred : {}'.format(clf_k_nearest_neighbors, clf_pred))

	if __name__ == '__main__':
		main()
