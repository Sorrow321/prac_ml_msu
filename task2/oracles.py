import numpy as np
from scipy import sparse


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

        
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """
    
    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        # В конце вычитается число - так я убираю регуляризацию для свободного члена
        
        return np.logaddexp(0, -np.multiply(y, X.dot(w))).sum() / X.shape[0] + self.l2_coef / 2 * (np.linalg.norm(w) ** 2 - w[-1] ** 2) 
        
    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        # В конце вычитается вектор - так я убираю регуляризацию для свободного члена
        return -X.T.dot(np.multiply(y, np.reciprocal(1 + np.exp(np.multiply(y, X.dot(w)))))) / X.shape[0] + self.l2_coef * (w - w[-1] * np.eye(1, len(w), len(w) - 1)[0])