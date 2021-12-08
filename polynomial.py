import math

import numpy as np
from data import polynomial_function
from prettytable import PrettyTable


def get_values(function, a, b, n):
    step = (b - a) / n

    x = [a + i * step for i in range(n + 1)]
    y = [function(a + i * step) for i in range(n + 1)]

    return x, y


def print_values_table(x, y):
    table = PrettyTable(['x', *x])
    table.add_row(['y', *y])
    print(table)


def lagrange_polynomial(x_point, x, y, n):
    y_point = 0
    for i in range(n):
        product = 1
        for j in range(n):
            if i != j:
                product = product * (x_point - x[j]) / (x[i] - x[j])
        y_point = y_point + product * y[i]

    return y_point


def finite_difference(y, i, k):
    if k == 0:
        return y[i]
    if k == 1:
        return y[i + 1] - y[i]

    return finite_difference(y, i + 1, k - 1) - finite_difference(y, i, k - 1)


def newton_polynomial(x_point, x, y, n, step):
    y_point = y[0]
    for i in range(1, n):
        product = 1
        delta = finite_difference(y, 0, i)
        for j in range(i):
            product = product * (x_point - x[j])
        y_point = y_point + (delta * product) / (math.factorial(i) * step ** i)

    return y_point


def interpolation_error(x_point, x, y, n):
    product = np.absolute(np.max(y))
    for i in range(n):
        product = product * (x_point - x[i])

    return np.absolute(product / math.factorial(n))


def print_final_polynomial_table(function, x, y, n, step):
    header = ['xi', 'f(xi)', 'L(xi)', 'P(xi)', '|f(xi) - L(xi)|', 'R(xi)']
    table = PrettyTable(header)

    for i in range(n * 2 - 1):
        x_point = x[0] + i * step / 2.0

        data = [
            x_point,
            function(x_point),
            lagrange_polynomial(x_point, x, y, n),
            newton_polynomial(x_point, x, y, n, step),
            function(x_point) - lagrange_polynomial(x_point, x, y, n),
            interpolation_error(x_point, x, y, n)
        ]

        table.add_row(data)

    print(table)


if __name__ == '__main__':
    a = 1
    b = 5
    n = 10
    step = (b - a) / n

    x, y = get_values(polynomial_function, a, b, n)
    print_values_table(x, y)
    print_final_polynomial_table(polynomial_function, x, y, n, step)
