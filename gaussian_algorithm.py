import numpy as np
from data import a, b, a_test, b_test


# Метод Гаусса: прямой ход
def forward(a, b):
    # Размерность матрицы
    n = len(a[0])

    for k in range(0, n):
        for i in range(k + 1, n):
            r = a[i][k] / a[k][k]
            for j in range(k, n):
                a[i][j] = a[i][j] - r * a[k][j]
            b[i] = b[i] - b[k] * r

    return a, b


# Метод Гаусса: обратный ход
def backward(a, b):
    # Размерность матрицы
    n = len(a[0])


if __name__ == '__main__':
    print('Матрица коэффициентов:\n', a_test)
    matrix = gaussian_forward(a_test, b_test)
    print('\nМатрица коэффициентов после прямого хода:\n', matrix[0])
    print('Столбец свободных членов:\n', matrix[1])
