import numpy as np
from gaussian_algorithm import forward
from data import a_test2


def determinant(a):
    """
    Функция для вычисления определителя матрицы по схеме Гаусса.

    :param a: исходная матрица
    :return: определитель матрицы
    """
    # Некрасиво, зато не требует еще один аргумент и исходную функцию можно не трогать
    triangular, b = forward(a, np.zeros((len(a[0]), 1)))
    return np.trace(triangular)


def identity(n):
    """
    Функция создает единичную матрицу размера n.

    :param n: размерность матрицы
    :return: единичная матрица
    """
    a = np.zeros((n, n))

    for i in range(n):
        a[i][i] = 1

    return a


def inverse(a):
    """
    Функция вычисляет обратную матрицу методом Гаусса.

    :param a: исходная матрица
    :return: обратная матрица a^-1
    """
    # Размерность матрицы
    n = len(a[0])

    identity_matrix = identity(n)
    a, b = forward(a, identity_matrix, method='identity')

    return b
