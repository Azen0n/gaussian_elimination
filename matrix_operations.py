import numpy as np
import copy
from data import a_test, b_test, a_test2, b_test2
from norms import norm


def forward(a, b, method=None):
    """
    Функция приводит матрицу к треугольному виду, применяя метод Гаусса (схему единственного деления). Прямой ход.

    :param a: исходная матрица A
    :param b: исходный столбец свободных членов b
    :param method: single — схема единственного деления, identity — единичная матрица, pivot — с выбором главного элемента
    :return: матрица и столбец свободных членов после применения метода Гаусса
    """
    # Массив с методами, приводящими матрицу к треугольному виду
    triangular = np.array([None, 'single', 'identity'])

    a_copy = copy.deepcopy(a)
    b_copy = copy.deepcopy(b)

    # Размерность матрицы
    n = len(a_copy[0])

    if method in triangular:
        for k in range(0, n):                          # Проходим каждый столбец, одна итерация этого цикла зануляет его
            for i in range(k + 1, n):                  # Элемент под главной диагональю
                r = a_copy[i][k] / a_copy[k][k]
                for j in range(k, n):                               # Вычитаем из всей строки другую,
                    a_copy[i][j] = a_copy[i][j] - r * a_copy[k][j]  # k-й элемент равен нулю
                b_copy[i] = b_copy[i] - b_copy[k] * r               # То же самое со столбцом свободных членов

        if method == 'single':                         # Единицы на главной диагонали
            diagonal_ones(a_copy, b_copy)

        if method == 'identity':                       # Обратный цикл, после которого над главной диагональю
            for k in range(n - 1, -1, -1):             # тоже получаются нули
                for i in range(k - 1, -1, -1):
                    r = a_copy[i][k] / a_copy[k][k]
                    for j in range(n - 1, k - 1, -1):
                        a_copy[i][j] = a_copy[i][j] - r * a_copy[k][j]
                    b_copy[i] = b_copy[i] - b_copy[k] * r

            # В конце на главной диагонали устанавливаются нули,
            # после чего a_copy становится единичной матрицей, а b_copy — обратной матрицей исходной
            diagonal_ones(a_copy, b_copy)

    if method == 'pivot':
        for k in range(n):                      # Проходим все ~~столбцы~~ строки

            abs = np.absolute(a_copy)
            max = np.argmax(abs, axis=1)

            b_copy[k] /= a_copy[k][max[k]]           # Делим всю строку на максимальный элемент
            a_copy[k] /= a_copy[k][max[k]]

            for i in range(n):                  # Проходим все строки
                if i != k:                      # Пропуская строку с максимальным элементом
                    prod_b = b_copy[k] * a_copy[i][max[k]]       # Умножаем строку с максимальным элементом на k-тый
                    b_copy[i] -= prod_b                     # элемент и вычитаем ее из всей строки

                    prod = a_copy[k] * a_copy[i][max[k]]
                    a_copy[i] -= prod

    return a_copy, b_copy


def backward(a, b, method):
    """
    Функция вычисляет значения x СЛАУ Ax=b у матрицы треугольного вида. Обратный ход.

    :param a: матрица треугольного вида
    :param b: столбец свободных членов после применения метода Гаусса
    :param method: single — схема единственного деления, identity — единичная матрица, pivot — с выбором главного элемента
    :return: вектор со значениями x
    """
    # Размерность матрицы
    n = len(a[0])

    # Вектор со значениями x
    x = np.zeros(n)

    if method == 'pivot':                       # Единицы не на главной диагонали
        max = np.argmax(a, axis=1)
        for i in range(n):                      # Каждой единице присваивается соответствующее значение вектора b
            x[max[i]] = b[i]
    else:
        x[n - 1] = b[n - 1] / a[n - 1][n - 1]   # Вычисление x_n

        for i in range(n - 2, -1, -1):          # Вычисление остальных x, начиная с предпоследнего
            sum = 0
            for j in range(i + 1, n):           # В цикле умножаем уже вычисленные значения x на значения матрицы a
                sum += a[i][j] * x[j]
            x[i] = (b[i] - sum) / a[i][i]       # И делим на соответсвующее значение матрицы a на главной диагонали

    return x


def diagonal_ones(a, b):
    """
    Функция с помощью элементарных преобразований образует единицы на главной диагонали матрицы.
    **Функция меняет массивы напрямую, не создавая копии.**

    :param a: исходная матрица
    :param b: исходный столбец свободных членов
    """
    # Размерность матрицы
    n = len(a[0])

    diag = np.array([a[i][i] for i in range(n)])    # Запоминаем значения главной диагонали

    for i in range(0, n):                           # И делим на них каждую всю строку,
        for j in range(0, n):                       # чтобы получить единицу на главной диагонали
            a[i][j] = a[i][j] / diag[i]
        b[i] = b[i] / diag[i]


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


def gaussian(a, b, method=None, out=False):
    """
    Функция вычисляет значения x СЛАУ Ax=b, применяя метод Гаусса (схему единственного деления).
    В консоль выводятся исходные данные, промежуточные и финальные результаты.

    :param a: исходная матрица A
    :param b: исходный столбец свободных членов b
    :param method: single — схема единственного деления, identity — единичная матрица, pivot — с выбором главного элемента
    :param out: флаг для вывода процесса в консоль (по умолчанию False)
    :return: вектор со значениями x
    """
    a_forward, b_forward = forward(a, b, method=method)
    x = backward(a_forward, b_forward, method=method)

    if out:
        print('Исходная матрица коэффициентов:\n', a)
        print('Исходный столбец свободных членов:\n', b)
        print('\nМатрица коэффициентов после прямого хода:\n', a_forward)
        print('Столбец свободных членов:\n', b_forward)
        print('\nВектор со значениями x:\n', x)

    return x


def condition_number(a):
    """
    Функция находит числа обусловленности для матрицы a, применяя нормы 1 и 2.

    :param a: исходная матрица
    :return: числа обусловленности
    """
    inverted_a = inverse(a)
    norm1 = norm(a, p=1)
    norm1_inverted = norm(inverted_a, p=1)
    norm2 = norm(a, p=2)
    norm2_inverted = norm(inverted_a, p=2)

    condition_number1 = norm1 * norm1_inverted
    condition_number2 = norm2 * norm2_inverted

    return condition_number1, condition_number2


def calculate_error(x, x_delta, p):
    """
    Функция вычисляет относительную погрешность решения, применяя p-норму.

    :param x: исходный вектор (решение СЛАУ)
    :param x_delta: вектор с погрешностью
    :param p: значение p нормы
    :return: относительная погрешность решения
    """
    x_norm = norm(x, p=p)
    x_norm_delta = norm(x_delta, p=p)
    error = x_norm_delta / x_norm

    return error


def constant_terms_error(a, b, noise_value=0.01, method=None):
    """
    Функция вычисляет относительную погрешность решения и оценивает ее по формуле:

    .. math:: \\frac{\\|\\Delta x\\|}{\\|x\\|} \\leq \\nu (A) \\frac{\\|\\Delta b\\|}{\\|b\\|}

    Погрешность вносится в столбец свободных членов.

    :param a: исходная матрица
    :param b: исходный столбец свободных членов
    :param noise_value: значение погрешности (по умолчанию 0.01)
    :param method: метод Гаусса
    """
    b_new = copy.deepcopy(b)                        # Прибавление погрешности к столбцу свободных членов
    b_new[1][0] += noise_value

    x = gaussian(a, b, method=method)               # Решение без погрешности и с погрешностью
    x_new = gaussian(a, b_new, method=method)

    b_delta = np.subtract(b_new, b)                 # Вычисление разницы новых и старых значений
    x_delta = np.subtract(x_new, x)

    x_error1 = calculate_error(x, x_delta, p=1)     # Относительная погрешность решения вектора x, p = 1
    x_error2 = calculate_error(x, x_delta, p=2)     # Относительная погрешность решения вектора x, p = 1

    print('Относительная погрешность решения:')
    print('При p = 1: %.15f' % x_error1)
    print('При p = 2: %.15f' % x_error2)

    condition_number1, condition_number2 = condition_number(a)  # Числа обусловленности v(a)

    b_error1 = calculate_error(b, b_delta, p=1)     # Относительная погрешность решения вектора b, p = 1
    b_error2 = calculate_error(b, b_delta, p=2)     # Относительная погрешность решения вектора b, p = 1

    print('\nПогрешность: %s' % noise_value)
    print('Столбец свободных членов с погрешностью:')
    print(b_new)
    print('Решение с погрешностью:')
    print(x_new)

    if x_error1 <= condition_number1 * b_error1:
        print('\nПри норме p = 1 матрица исходной системы уравнений хорошо обусловлена: неравенство выполняется.')
    else:
        print('\nПри норме p = 1 матрица исходной системы уравнений является плохо обусловленной, так как неравенство '
              'не выполняется.')

    if x_error2 <= condition_number2 * b_error2:
        print('При норме p = 2 матрица исходной системы уравнений хорошо обусловлена: неравенство выполняется.')
    else:
        print('При норме p = 2 матрица исходной системы уравнений является плохо обусловленной, так как неравенство '
              'не выполняется.')


def coefficients_error(a, b, noise_value=0.01, method=None):
    """
    Функция вычисляет относительную погрешность решения.
    Погрешность вносится в матрицу значений A.

    :param a: исходная матрица
    :param b: исходный столбец свободных членов
    :param noise_value: значение погрешности (по умолчанию 0.01)
    :param method: метод Гаусса
    """
    a_new = copy.deepcopy(a)                        # Добавление погрешности к матрице коэффициентов
    a_new[0][0] += noise_value                      # (к первому элементу первой строки)

    x = gaussian(a, b, method=method)               # Решение без погрешности и с погрешностью
    x_new = gaussian(a_new, b, method=method)

    x_delta = np.subtract(x_new, x)                 # Вычисление разницы новых и старых значений

    x_error1 = calculate_error(x, x_delta, p=1)     # Относительная погрешность решения вектора x, p = 1
    x_error2 = calculate_error(x, x_delta, p=2)     # Относительная погрешность решения вектора x, p = 1

    print('\nПогрешность: %s' % noise_value)
    print('Матрица коэффициентов с погрешностью:')
    print(a_new)
    print('Решение с погрешностью:')
    print(x_new)

    print('\nОтносительная погрешность решения:')
    print('При p = 1: %.15f' % x_error1)
    print('При p = 2: %.15f' % x_error2)
