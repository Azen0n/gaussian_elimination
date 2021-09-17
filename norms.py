import numpy as np
from data import a, b


def norm(a, p):
    # Абсолютные значения матрицы
    abs = np.absolute(a)

    # P-норма при p = 1
    # Максимальная сумма столбцов
    if p == 1:
        sums = np.sum(abs, axis=0)
        answer = np.max(sums)
    # P-норма при p = inf
    # Максимальная сумма строк
    elif p == 'inf':
        sums = np.sum(abs, axis=1)
        answer = np.max(sums)
    # Норма Фробениуса (по умолчанию), P-норма при p = 2
    else:
        answer = np.sqrt(np.sum(np.square(abs)))

    return answer


# Вычисление приколов для приколов
def residuals(a, b, x):
    # Размерность матрицы
    n = len(a[0])

    sum = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            sum[i] += a[i][j] * x[j]

    res = np.zeros((n, 1))
    for i in range(n):
        res[i] = b[i] - sum[i]

    print('\nНевязки:')
    print('||X||1 = ', norm(res, 1))
    print('||X||2 = ', norm(res, 2))
    print('||X||inf = ', norm(res, 'inf'))

