import numpy as np
from prettytable import prettytable

from polynomial import get_values, print_values_table


def euler(f, initial, x, a, b, n):
    """Решение дифференциального уравнения y' = f(x, y), где y(0) = initial методом Эйлера"""
    h = (b - a) / n
    y = np.zeros(len(x))
    y[0] = initial
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    return y


def euler_cauchy(f, initial, x, a, b, n):
    """Решение дифференциального уравнения y' = f(x, y), где y(0) = initial методом Эйлера–Коши"""
    h = (b - a) / n
    y = np.zeros(len(x))
    y[0] = initial
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h / 2 * (f(x[i], y[i]) + f(x[i] + h, y[i] + h * f(x[i], y[i])))
    return y


def half_euler_cauchy(f, initial, x, a, b, n):
    """Решение дифференциального уравнения y' = f(x, y), где y(0) = initial методом Эйлера–Коши на полуцелой сетке"""
    h = (b - a) / n
    y = np.zeros(len(x))
    y[0] = initial
    for i in range(len(x) - 1):
        y[i + 1] = y[i] + h * f(x[i] + h / 2, y[i] + h / 2 * f(x[i], y[i]))
    return y


def runge_kutta(f, initial, x, a, b, n):
    """Решение дифференциального уравнения y' = f(x, y), где y(0) = initial методом Рунге–Кутты"""
    h = (b - a) / n
    y = np.zeros(len(x))
    y[0] = initial
    for i in range(len(x) - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + k1 * h / 2)
        k3 = f(x[i] + h / 2, y[i] + k2 * h / 2)
        k4 = f(x[i] + h, y[i] + k3 * h)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def print_methods_table(x, y, y1, y2, y3, y4):
    header = ['xi', 'Точное', 'Метод Эйлера', 'Метод Эйлера—Коши', 'Метод Эйлера—Коши 0.5h', 'Метод Рунге—Кутты']
    table = prettytable.PrettyTable(header)
    for i in range(len(y1)):
        table.add_row([x[i], y[i], y1[i], y2[i], y3[i], y4[i]])
    print(table)


def function(x):
    return np.exp(np.exp(x) - 1)


def function_prime(x, y):
    return y * np.exp(x)


def main():
    a = 0
    b = 1
    n = 10
    initial = function(a)
    x, y = get_values(function, a, b, n)
    print_values_table(x, y)

    y1 = euler(function_prime, initial, x, a, b, n)
    y2 = euler_cauchy(function_prime, initial, x, a, b, n)
    y3 = half_euler_cauchy(function_prime, initial, x, a, b, n)
    y4 = runge_kutta(function_prime, initial, x, a, b, n)

    print_methods_table(x, y, y1, y2, y3, y4)


if __name__ == '__main__':
    main()
