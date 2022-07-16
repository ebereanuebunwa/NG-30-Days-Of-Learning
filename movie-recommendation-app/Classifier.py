import numpy as np
from operator import itemgetter


class KNearestNeighbours:

    def __init__(self, data, target, test_point, k):
        self.data = data
        self.target = target
        self.test_point = test_point
        self.k = k  # Number of neighbours to consider

        self.distances = list()
        self.categories = list()
        self.indices = list()
        self.counts = list()
        self.category_assigned = None  # Assigned category

    @staticmethod
    def dist(p1, p2):
        """
        Calculate the euclidean distance between two points
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))  # Calculate the euclidean distance between two points

    def fit(self):
        """
        Method that performs the KNN Classification
        """

        # Create a list of distances and a list of categories in tuples (distance, category)
        # from test point to all points in the data
        self.distances.extend(
            [(self.dist(self.test_point, point), i) for point, i in zip(self.data, [i for i in range(len(self.data))])])

        # Sort the list of distances and categories in ascending order
        sorted_list = sorted(self.distances, key=itemgetter(0))

        # Get the categories of the k nearest neighbours
        # Fetch the indices of the k nearest neighbours from the data
        self.indices.extend([index for (val, index) in sorted_list[:self.k]])

        # Fetch the categories of the k nearest neighbours from the target
        for i in self.indices:
            self.categories.append(self.target[i])

        # Count the number of times each category appears in the list of categories
        self.counts.extend([(i, self.categories.count(i)) for i in set(self.categories)])

        # Get the category with the highest count
        self.category_assigned = sorted(self.counts, key=itemgetter(1), reverse=True)[0][0]
