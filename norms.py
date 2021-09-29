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

    if p == 1:                          # P-норма при p = 1
        sums = np.sum(abs, axis=0)      # Максимальная сумма столбцов
        answer = np.max(sums)

    elif p == 'inf':                    # P-норма при p = inf
        sums = np.sum(abs, axis=1)      # Максимальная сумма строк
        answer = np.max(sums)

    else:                               # Норма Фробениуса (по умолчанию), P-норма при p = 2
        answer = np.sqrt(np.sum(np.square(abs)))

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

    print('\nНевязки:')
    print('Манхэттенская норма (p = 1): %.15f' % norm(res, 1))
    print('Норма Фробениуса (p = 2): %.15f' % norm(res, 2))
    print('Максимальная норма (p = inf): %.15f' % norm(res, 'inf'))
