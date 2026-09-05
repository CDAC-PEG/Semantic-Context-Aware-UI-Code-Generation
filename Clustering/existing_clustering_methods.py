import time
import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances


class Existing_DBSCAN:
    model_name = "DBSCAN"

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, X):
        start = time.perf_counter()

        model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )

        labels = model.fit_predict(X)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return labels, elapsed_ms


class Existing_KMeans:
    model_name = "K-Means"

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster(self, X):
        start = time.perf_counter()

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        labels = model.fit_predict(X)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return labels, elapsed_ms


class Existing_FFC:
    """
    Farthest First Clustering.

    Selects the first centroid, then repeatedly chooses
    the point farthest from its nearest selected centroid.
    """

    model_name = "Farthest First Clustering"

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(
            self.random_state
        )

        first_index = rng.integers(
            0,
            len(X)
        )

        centroids = [
            X[first_index]
        ]

        while len(centroids) < self.n_clusters:

            distances = pairwise_distances(
                X,
                np.vstack(centroids),
                metric="euclidean"
            )

            nearest_distance = np.min(
                distances,
                axis=1
            )

            next_index = int(
                np.argmax(
                    nearest_distance
                )
            )

            centroids.append(
                X[next_index]
            )

        return np.vstack(
            centroids
        )

    def cluster(self, X):
        start = time.perf_counter()

        centroids = self._initialize_centroids(
            X
        )

        distances = pairwise_distances(
            X,
            centroids,
            metric="euclidean"
        )

        labels = np.argmin(
            distances,
            axis=1
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return labels, elapsed_ms


class Existing_FCM:
    """
    Fuzzy C-Means clustering implemented locally
    to avoid requiring an additional package.

    Final crisp labels are obtained from the highest
    membership value for silhouette-score evaluation.
    """

    model_name = "Fuzzy C-Means"

    def __init__(
        self,
        n_clusters=3,
        m=2.0,
        max_iter=150,
        tol=1e-5,
        random_state=42
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def cluster(self, X):
        start = time.perf_counter()

        X = np.asarray(
            X,
            dtype=float
        )

        n_samples = X.shape[0]

        rng = np.random.default_rng(
            self.random_state
        )

        U = rng.random(
            (
                n_samples,
                self.n_clusters
            )
        )

        U = U / np.sum(
            U,
            axis=1,
            keepdims=True
        )

        for _ in range(
            self.max_iter
        ):
            U_old = U.copy()

            Um = U ** self.m

            centers = (
                Um.T @ X
            ) / (
                np.sum(
                    Um,
                    axis=0
                )[:, None]
                + 1e-12
            )

            distances = pairwise_distances(
                X,
                centers,
                metric="euclidean"
            )

            distances = np.maximum(
                distances,
                1e-12
            )

            power = (
                2.0
                / (
                    self.m - 1.0
                )
            )

            for i in range(
                n_samples
            ):
                for j in range(
                    self.n_clusters
                ):
                    ratio = (
                        distances[i, j]
                        / distances[i]
                    )

                    U[i, j] = (
                        1.0
                        / np.sum(
                            ratio ** power
                        )
                    )

            if np.linalg.norm(
                U - U_old
            ) < self.tol:
                break

        labels = np.argmax(
            U,
            axis=1
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return labels, elapsed_ms
