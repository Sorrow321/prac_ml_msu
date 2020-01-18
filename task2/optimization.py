import numpy as np
import time
import scipy
from oracles import BinaryLogistic
from scipy.special import expit


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter

        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**kwargs)
        
    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array

        ВАЖНО! Вектор y должен состоять из 1 и -1, а не 1 и 0.
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        # initial value
        self.w = w_0 if not w_0 is None else np.zeros(X.shape[1])

        loss_value = self.oracle.func(X, y, self.w)
        history = {'time': [], 'func': [loss_value], 'accuracy': [(y == self.predict(X)).sum() / len(y)]}
        prev_time = time.time()

        for i in range(1, self.max_iter + 1):
            new_w = self.w - self.step_alpha / (i ** self.step_beta) * self.oracle.grad(X, y, self.w)
            new_loss_value = self.oracle.func(X, y, new_w)

            if trace:
                history['func'].append(new_loss_value)
                history['time'].append(time.time() - prev_time)
                history['accuracy'].append((y == self.predict(X)).sum() / len(y))
                prev_time = time.time()

            if abs(loss_value - new_loss_value) < self.tolerance:
                self.w = new_w
                break
            
            loss_value = new_loss_value
            self.w = new_w
        
        if trace:
            return history

        
    def predict(self, X, threshold=0.5):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        result = (self.predict_proba(X) > threshold).astype('int')
        result[result == 0] = -1
        return result

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        return expit(X.dot(self.w))
        
    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.oracle.func(X, y, self.w)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        
        max_iter - максимальное число итераций (эпох)
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_epoch = max_iter
        self.batch_size = batch_size
        np.random.seed(random_seed)

        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**kwargs)
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        ВАЖНО! Вектор y должен состоять из 1 и -1, а не 1 и 0.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """

        # initial value
        self.w = w_0 if not w_0 is None else np.zeros(X.shape[1])
        
        if isinstance(X, scipy.sparse.coo.coo_matrix):
            X = X.tocsr()
        
        loss_value = self.oracle.func(X, y, self.w)
        history = {'epoch_num': [0], 'time': [], 'func': [loss_value], 'weights_diff': [0], 'accuracy': [(y == self.predict(X)).sum() / len(y)]}
        prev_time = time.time()
        iter_id = 1
        calc = time.time() - time.time()
        for epoch_i in range(1, self.max_epoch + 1): 
            permutation = np.random.permutation(X.shape[0])
            
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            self.w_prev = np.copy(self.w)
            for batch_i in range(int(np.ceil(X.shape[0] / self.batch_size))):
            
                X_batch = X_shuffled[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]
                y_batch = y_shuffled[batch_i * self.batch_size : (batch_i + 1) * self.batch_size]
    
                self.w = self.w - self.step_alpha / (iter_id ** self.step_beta) * self.oracle.grad(X_batch, y_batch, self.w)
                iter_id += 1
            
            new_loss_value = self.oracle.func(X, y, self.w)
            
            
            if trace:
                history['epoch_num'].append(epoch_i)
                history['time'].append(time.time() - prev_time)
                history['func'].append(new_loss_value)
                history['accuracy'].append((y == self.predict(X)).sum() / len(y))
                diff = self.w - self.w_prev
                history['weights_diff'].append(np.dot(diff, diff))
                prev_time = time.time()

            if abs(loss_value - new_loss_value) < self.tolerance:
                break
            loss_value = new_loss_value

        
        if trace:
            return history

    def predict(self, X, threshold=0.5):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        result = (self.predict_proba(X) > threshold).astype('int')
        result[result == 0] = -1
        return result

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        return expit(X.dot(self.w))
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.w