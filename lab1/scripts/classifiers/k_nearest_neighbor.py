import numpy as np


class KNearestNeighbor:

    def train(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)

    def compute_distances_two_loops(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.empty((len(X), len(self.X_train)))
        for i, test_image in enumerate(X):
            for j, train_image in enumerate(self.X_train):
                diff = test_image - train_image
                distances[i, j] = np.sqrt(diff @ diff)
        return distances

    def compute_distances_one_loop(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.empty((len(X), len(self.X_train)))
        for i, test_image in enumerate(X):
            distances[i] = np.linalg.norm(self.X_train - test_image, axis=1)
        return distances

    def compute_distances_no_loops(self, X):
        X = np.asarray(X, dtype=np.float64)
        test_sq = np.sum(X * X, axis=1)[:, None]
        train_sq = np.sum(self.X_train ** 2, axis=1)
        squared = test_sq + train_sq - 2 * (X @ self.X_train.T)
        return np.sqrt(np.maximum(squared, 0))

    def compute_distances(self, X, num_loops=0):
        methods = {0: self.compute_distances_no_loops,
                   1: self.compute_distances_one_loop,
                   2: self.compute_distances_two_loops}
        if num_loops not in methods:
            raise ValueError("num_loops должен быть 0, 1 или 2")
        return methods[num_loops](X)

    def predict_labels(self, dists, k=1):
        if not 1 <= k <= len(self.y_train):
            raise ValueError("k вне диапазона обучающей выборки")
        nearest = np.argsort(dists, axis=1, kind="stable")[:, :k]
        predictions = []
        for neighbors in nearest:
            labels, votes = np.unique(self.y_train[neighbors], return_counts=True)
            predictions.append(labels[votes.argmax()])
        return np.asarray(predictions, dtype=self.y_train.dtype)

    def predict(self, X, k=1, num_loops=0):
        return self.predict_labels(self.compute_distances(X, num_loops), k)
