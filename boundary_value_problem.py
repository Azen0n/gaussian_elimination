import numpy as np
from polynomial import get_values
from matrix_operations import gaussian
from prettytable import prettytable
from matplotlib import pyplot as plt


def finite_difference(function, a, b, n, h, p, q, f, c, d):
    """
    Решение краевой задачи методом конечных разностей.

    :param function: точное решение краевой задачи, функция y(x)
    :param a, b: отрезок [a, b], где a = x0 и b = xn
    :param n: количество интервалов
    :param h: шаг (расстояние между узлами)
    :param x: множество точек X (узлы)
    :param p, q, f: функциональные коэффициенты диффернециального уравнения, p = p(x), q = q(x) и f = f(x)
    :param c, d: коэффициенты краевых условий в точках a и b соответственно
    :return: приближенное решение краевой задачи, вектор y, где yi ≈ y(xi)
    """
    x, y = get_values(function, a, b, n)
    a, b = generate_tridiagonal_matrix(h, x, p, q, f, c, d)
    y2 = gaussian(a, b, method='tridiagonal')
    return y2


def print_approximation(x, y, y2):
    table = prettytable.PrettyTable(['x', 'y(x) точное', 'y(x) приближенное'])
    for i in range(len(x)):
        table.add_row([x[i], y[i], y2[i]])
    print(table)


def plot_approximation(x, y, y2):
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.title('Приближенное решение краевой задачи')
    plt.legend(['Точное решение', 'Приближенное решение'])
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.show()


def generate_tridiagonal_matrix(h, x, p, q, f, c, d):
    """
    Трехдиагональная матрица для дальнейшего приближенного решения краевой задачи.

    :param h: шаг (расстояние между узлами)
    :param x: множество точек X (узлы)
    :param p, q, f: функциональные коэффициенты диффернециального уравнения, p = p(x), q = q(x) и f = f(x)
    :param c, d: коэффициенты краевых условий в точках a и b соответственно
    :return: матрица коэффициентов A (matrix) и столбец свободных членов b уравнения Ay=b
    """

    # Коэффициенты разностного уравнения
    C = [2 + h * p(x[i]) - h * h * q(x[i]) for i in range(len(x))]
    B = [1 + h * p(x[i]) for i in range(len(x))]
    F = [- h * h * f(x[i]) for i in range(len(x))]

    # Коэффициенты разностного уравнения (на основе краевых условий)
    chi1 = c[1] / (c[1] - h * c[0])
    chi2 = d[1] / (d[1] + h * d[0])
    matrix = np.zeros((len(x), len(x)))
    b = np.zeros((len(x), 1))

    for i in range(1, len(x) - 1):
        matrix[i][i - 1] = 1
        matrix[i][i] = - C[i]
        matrix[i][i + 1] = B[i]
        b[i][0] = F[i]

    b[0][0] = - c[2] * h / (c[1] - h * c[2])
    b[-1][0] = d[2] * h / (d[1] + h * d[0])
    matrix[0][0] = 1
    matrix[0][1] = - chi1
    matrix[-1][-1] = 1
    matrix[-1][-2] = - chi2

    return matrix, b


def function(x):
    return x + 1 + 1 / x


def p(x):
    return (x + 1) / (x ** 2 + 0.5 * x)


def q(x):
    return - 2 / (2 * x ** 2 + x)


def f(x):
    return 0


def main():
    # Краевая задача:
    # x * (2 * x + 1) * y'' + 2 * (x + 1) * y' - 2 * y = 0
    # y'(1) = 0
    # y(3) - y'(3) = 31 / 9
    # Для выражения p(x) и q(x) дифференциальное уравнение было разделено на x * (2 * x + 1)
    a = 1
    b = 3
    c = [0, 1, 0]
    d = [1, -1, 31 / 9]

    n = 30
    h = (b - a) / n

    x, y = get_values(function, a, b, n)
    y2 = finite_difference(function, a, b, n, h, p, q, f, c, d)
    print_approximation(x, y, y2)
    plot_approximation(x, y, y2)


if __name__ == '__main__':
    main()
