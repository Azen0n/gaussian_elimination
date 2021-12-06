import numpy as np
from prettytable import PrettyTable

from data import polynomial_function
from polynomial import print_values_table
from matrix_operations import gaussian


def cubic_spline(x_point, x, y, n, h):
    """Возвращает значение кубического сплайна дефекта 1 в точке x_point"""
    a = np.array(y)
    a_tridiagonal, b_tridiagonal = generate_tridiagonal_matrix(x, y, h)
    c = gaussian(a_tridiagonal, b_tridiagonal, method='pivot')

    d = [(c[i + 1] - c[i]) / 3.0 * h for i in range(n)]
    d.append(-c[-1] / 3.0 * h)

    b = [((y[i + 1] - y[i]) / h) - (h / 3.0 * (c[i + 1] + 2.0 * c[i])) for i in range(n)]
    b.append(((y[-1] - y[-2]) / h) - (h * c[-1] * 2.0) / 3.0)

    i = get_x_index(x_point, x)
    spline = a[i] + b[i] * (x_point - x[i]) + c[i] * (x_point - x[i]) ** 2 + d[i] * (x_point - x[i]) ** 3

    return spline


def get_x_index(x_point, x):
    for i in range(1, len(x)):
        if x_point == x[0]:
            return 0
        elif x[i] >= x_point >= x[i - 1]:
            return i
    raise IndexError('Index out of range')


def generate_tridiagonal_matrix(x, y, h):
    """Возвращает матрицу коэффициентов A и столбец свободных членов b уравнения Ac=b"""
    n = len(x)
    a = np.zeros((n, n))
    b = np.zeros((n, 1))

    a[0][0] = 1                     # Значение шага узлов интерполяции одинаково,
    a[-1][-1] = 1                   # поэтому исходное уравнение разделено на h
    for i in range(1, n - 1):
        a[i][i - 1] = 1
        a[i][i] = 4
        a[i][i + 1] = 1
        b[i] = 3.0 / (h * h) * (y[i - 1] - 2 * y[i] + y[i + 1])

    return a, b


def print_spline_table(function, x, y, n, h):
    header = ['xi', 'f(xi)', 'S(xi)', '|f(xi) - S(xi)|']
    table = PrettyTable(header)

    for i in range(n * 2 + 1):
        x_point = np.round(x[0] + i * h / 2.0, 1)
        data = [x_point, function(x_point), cubic_spline(x_point, x, y, n, h),
                np.absolute(function(x_point) - cubic_spline(x_point, x, y, n, h))]
        table.add_row(data)

    print(table)


def main():
    a = 1
    b = 5
    n = 5
    h = (b - a) / n
    x, y = print_values_table(polynomial_function, a, b, n)
    print_spline_table(polynomial_function, x, y, n, h)


if __name__ == '__main__':
    main()
