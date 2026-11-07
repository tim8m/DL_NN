import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """Обучение классификатора - просто запоминаем данные"""
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """Предсказание меток для X используя k ближайших соседей"""
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        """Вычисление расстояний от каждой точки X до всех точек обучающей выборки"""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # Вычисляем евклидовы расстояния используя формулу: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        X_train_norm = np.sum(self.X_train**2, axis=1, keepdims=True)
        dists = np.sqrt(X_norm + X_train_norm.T - 2 * np.dot(X, self.X_train.T))

        return dists

    def predict_labels(self, dists, k=1):
        """Предсказание меток на основе матрицы расстояний"""
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # Находим k ближайших соседей
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            # Выбираем наиболее часто встречающуюся метку
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
