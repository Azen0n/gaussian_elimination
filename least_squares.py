import numpy as np
from prettytable import PrettyTable

from data import polynomial_function
from polynomial import print_values_table
from matrix_operations import gaussian


def least_squares_coefficients(x, y, m):
    """Вычисление коэффициентов a аппроксимирующей функции."""
    f = np.zeros(2 * m + 1)
    t = np.zeros((m + 1, 1))

    for k in range(2 * m + 1):
        f[k] = sum([x[i] ** k for i in range(len(x))])

    for k in range(m + 1):
        t[k] = sum([x[i] ** k * y[i] for i in range(len(x))])

    matrix = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            matrix[i][j] = f[i + j]

    a = gaussian(matrix, t, method='pivot')

    return a


def least_squares(x, a):
    """Значение аппроксимирующей функции в точке x."""
    return a[0] + sum([a[i] * x ** i for i in range(1, len(a))])


def print_least_squares_table(function, x, coefficients, n, h):
    header = ['xi', 'f(xi)', 'Q(xi)', '|f(xi) - Q(xi)|']
    table = PrettyTable(header)

    for i in range(n * 2 + 1):
        x_point = np.round(x[0] + i * h / 2.0, 1)

        data = [
            x_point,
            function(x_point),
            least_squares(x_point, coefficients),
            np.absolute(function(x_point) - least_squares(x_point, coefficients))
        ]

        table.add_row(data)

    print(table)


def print_std_table(x, y, m):
    header = ['m', 'ср.кв. откл. |f(xi) - Q(xi)|']
    table = PrettyTable(header)
    for i in range(1, m):
        coefficients = least_squares_coefficients(x, y, i)
        y_new = [least_squares(i, coefficients) for i in x]
        delta = np.absolute(np.subtract(y_new, y))
        table.add_row([i, np.std(delta)])

    print(table)


def main():
    a = 1
    b = 5
    n = 5
    h = (b - a) / n
    x, y = print_values_table(polynomial_function, a, b, n)

    m = 3
    coefficients = least_squares_coefficients(x, y, m)
    print_least_squares_table(polynomial_function, x, coefficients, n, h)
    print_std_table(x, y, n + 1)


if __name__ == '__main__':
    main()
