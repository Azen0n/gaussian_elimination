import numpy as np
from gaussian_algorithm import forward


def determinant(a):
    """
    Функция для вычисления определителя матрицы по схеме Гаусса.

    :param a: исходная матрица
    :return: определитель матрицы
    """
    # Некрасиво, зато не требует еще один аргумент и исходную функцию можно не трогать
    triangular, b = forward(a, np.zeros((len(a[0]), 1)))
    return np.trace(triangular)
