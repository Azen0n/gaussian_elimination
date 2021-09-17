import numpy as np


# Метод Гаусса: прямой ход
def forward(a, b):
    # Размерность матрицы
    n = len(a[0])

    for k in range(0, n):                           # Проходим каждый столбец, одна итерация этого цикла зануляет его
        for i in range(k + 1, n):                   # Элемент под главной диагональю
            r = a[i][k] / a[k][k]
            for j in range(k, n):                   # Вычитаем из всей строки другую, k-й элемент равен нулю
                a[i][j] = a[i][j] - r * a[k][j]
            b[i] = b[i] - b[k] * r                  # То же самое со столбцом свободных членов

    diag = np.array([a[i][i] for i in range(n)])    # Запоминаем значения главной диагонали

    for i in range(0, n):                           # И делим на них каждую всю строку,
        for j in range(0, n):                       # чтобы получить единицу на главной диагонали
            a[i][j] = a[i][j] / diag[i]
        b[i] = b[i] / diag[i]

    return a, b


# Метод Гаусса: обратный ход
def backward(a, b):
    # Размерность матрицы
    n = len(a[0])

    # Вектор со значениями x
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / a[n - 1][n - 1]   # Вычисление x_n

    for i in range(n - 2, -1, -1):          # Вычисление остальных x, начиная с предпоследнего
        sum = 0
        for j in range(i + 1, n):           # В цикле умножаем уже вычисленные значения x на значения матрицы a
            sum += a[i][j] * x[j]
        x[i] = (b[i] - sum) / a[i][i]       # И делим на соответсвующее значение матрицы a на главной диагонали

    return x


# Функция для импорта
def gaussian(a, b):
    print('Исходная матрица коэффициентов:\n', a)
    print('Исходный столбец свободных членов:\n', b)

    forward_result = forward(a, b)
    print('\nМатрица коэффициентов после прямого хода:\n', forward_result[0])
    print('Столбец свободных членов:\n', forward_result[1])

    x = backward(forward_result[0], forward_result[1])
    print('\nВектор со значениями x:\n', x)

    return x
