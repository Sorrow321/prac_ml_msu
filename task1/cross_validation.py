import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    my_range = np.arange(n)
    result = []
    fold_size = int(np.ceil(n / n_folds))
    for k in range(n_folds):
        train_1 = np.arange(0, k * fold_size)
        train_2 = np.arange((k + 1) * fold_size, n)
        val = np.arange(k * fold_size, min((k + 1) * fold_size, n))
        result.append( (np.concatenate((train_1, train_2)), val) )
    return result

def _accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / y_pred.shape[0] 

def _most_frequent_value(x):
    (values,counts) = np.unique(x, return_counts=True, axis=0)
    ind = np.argmax(counts)
    return values[ind]

def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    if not cv:
        cv = kfold(X.shape[0], 2)
    result = {}
    
    max_k = k_list[len(k_list) - 1]
    for i, data in enumerate(cv):
        train = data[0]
        test = data[1]
        
        clf = KNNClassifier(k=max_k, **kwargs)
        clf.fit(X[train], y[train])
        
        if clf.weights:
            dist, idx = clf.find_kneighbors(X[test], True)
            nearest_labels = y[train][idx]
            
            scores = [{} for j in range(dist.shape[0])]
            for k in range(1, max_k + 1):
                answers = np.zeros(dist.shape[0], dtype=np.int32)
                for j in range(dist.shape[0]):
                    if nearest_labels[j, k - 1] not in scores[j].keys():
                        scores[j][nearest_labels[j, k - 1]] = 0
                    scores[j][nearest_labels[j, k - 1]] += 1 / (dist[j, k - 1] + 1e-5)
                    
                    answers[j] = max(scores[j], key=scores[j].get)
                
                if k in k_list:
                    if k not in result.keys():
                        result[k] = np.zeros(len(cv))
                    result[k][i] = _accuracy(answers, y[test])
        else:
            idx = clf.find_kneighbors(X[test], False)
            nearest_labels = y[train][idx]
            
            scores = [{} for j in range(test.shape[0])]
            for k in range(1, max_k + 1):
                answers = np.zeros(test.shape[0], dtype=np.int32)
                for j in range(test.shape[0]):
                    if nearest_labels[j, k - 1] not in scores[j].keys():
                        scores[j][nearest_labels[j, k - 1]] = 0
                    scores[j][nearest_labels[j, k - 1]] += 1
                    
                    answers[j] = max(scores[j], key=scores[j].get)
                
                if k in k_list:
                    if k not in result.keys():
                        result[k] = np.zeros(len(cv))
                    result[k][i] = _accuracy(answers, y[test])
    return result