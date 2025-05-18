import random
import math
import matplotlib as plt


class Kmeans:
    def __init__(self, points, k, max_iterations=100):
        self.points = points
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = random.sample(self.points, self.k)
        self.assignments = []
        self.iterations = 0
        self.first_centroids = self.centroids[:]\

    def run(self):
        pass