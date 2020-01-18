import numpy as np


# found this brief and efficient code in this article
# https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
def euclidean_distance(X, Y):
    return np.sqrt(-2 * np.dot(X, Y.T) + np.sum(X ** 2, axis=1)[:, np.newaxis]\
    + np.sum(Y ** 2, axis = 1).T)

def cosine_distance(X, Y):
    return 1 - np.dot(X, Y.T) / np.dot(np.linalg.norm(X, axis=1)[:, None],\
    np.linalg.norm(Y, axis=1)[:, None].T)

class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.strategy = strategy
        self.metric = metric
        self.k = k
        self.weights = weights
        self.test_block_size = test_block_size
        if self.strategy != 'my_own':
            from sklearn.neighbors import NearestNeighbors
            self.clf = NearestNeighbors(algorithm=self.strategy, metric=self.metric, n_neighbors=k)
    
    def fit(self, X, y):
        if self.strategy == 'my_own':
            self.data = X
            self.labels = y
        else:
            self.clf.fit(X)
            self.labels = y
        
    def _most_frequent_value(self, x):
        (values,counts) = np.unique(x, return_counts=True, axis=0)
        ind = np.argmax(counts)
        return values[ind]
    
    def find_kneighbors(self, X, return_distance):
        idx_l = 0
        idx_r = idx_l + self.test_block_size
        
        if return_distance:
            result = (np.zeros((X.shape[0], self.k), dtype=np.float64), np.zeros((X.shape[0], self.k), dtype=np.int64))
        else:
            result = np.zeros((X.shape[0], self.k), dtype=np.int64)
        
        while idx_l < X.shape[0]:
            result_local = self._find_kneighbors(result, idx_l, idx_r, X[idx_l : idx_r], return_distance)
            idx_l += self.test_block_size
            idx_r = idx_l + self.test_block_size
        return result
    
    def _find_kneighbors(self, target, idx_l, idx_r, X, return_distance):
        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                pairwise_dist = euclidean_distance(self.data, X)  
            elif self.metric == 'cosine':
                pairwise_dist = cosine_distance(self.data, X)
            else:
                raise Exception('Invalid metric')
            
            nearest = pairwise_dist.argsort(axis=0).T[:, :self.k]
            if return_distance:
                target[0][idx_l : idx_r] = pairwise_dist.T[np.arange(pairwise_dist.shape[1])[:, None], nearest]
                target[1][idx_l : idx_r] = nearest
            else:
                target[idx_l : idx_r] = nearest
        else:
            temp = self.clf.kneighbors(X, return_distance=return_distance)
            if return_distance:
                target[0][idx_l : idx_r] = temp[0]
                target[1][idx_l : idx_r] = temp[1]
            else:
                target[idx_l : idx_r] = temp
    
    def predict(self, X):
        if self.weights:
            dist, idx = self.find_kneighbors(X, True)
            nearest_labels = self.labels[idx]
            result = np.zeros(dist.shape[0], dtype=np.int32)
            for i in range(dist.shape[0]):
                scores = {}
                for j in range(self.k):
                    if nearest_labels[i, j] not in scores.keys():
                        scores[nearest_labels[i, j]] = 0
                    scores[nearest_labels[i, j]] += 1 / (dist[i, j] + 1e-5)
                result[i] = max(scores, key=scores.get)
            return result
        else:
            nearest_labels = self.labels[self.find_kneighbors(X, False)]
            return np.apply_along_axis(self._most_frequent_value, 1, nearest_labels)