import numpy as np


def norm(a, p):
    """
    Функция вычисления нормы матрицы.

    :param a: матрица
    :param p: значение для P-нормы
    :return: норма матрицы
    """
    # Абсолютные значения матрицы
    abs = np.absolute(a)

    if p == '1':                        # P-норма при p = 1
        if a.ndim == 1:
            answer = np.max(abs)
        else:
            sums = np.sum(abs, axis=1)  # Максимальная сумма столбцов
            answer = np.max(sums)

    elif p == '2':                      # Норма Фробениуса, P-норма при p = 2
        answer = np.sqrt(np.sum(np.square(abs)))

    elif p == 'inf':                    # P-норма при p = inf
        sums = np.sum(abs, axis=0)      # Максимальная сумма строк
        answer = np.max(sums)

    return answer


def residuals(a, b, x):
    """
    Функция вычисления невязок.

    :param a: исходная матрица
    :param b: исходный столбец свободных членов
    :param x: вычисленные значения x СЛАУ
    """
    # Размерность матрицы
    n = len(a[0])

    sum = np.zeros((n, 1))
    for i in range(n):                  # Подставляем x в уравнения
        for j in range(n):
            sum[i] += a[i][j] * x[j]

    res = np.zeros((n, 1))
    for i in range(n):                  # Вычитаем из столбца свободных членов
        res[i] = b[i] - sum[i]

    print('Манхэттенская норма (p = 1): %.15e' % norm(res, '1'))
    print('Норма Фробениуса (p = 2): %.15e' % norm(res, '2'))
    print('Максимальная норма (p = inf): %.15e' % norm(res, 'inf'))
