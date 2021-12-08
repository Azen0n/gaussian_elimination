import numpy as np
from prettytable import PrettyTable
import scipy.integrate as integrate

from polynomial import get_values, print_values_table


# Вычислить интеграл по формулам:
# 1. прямоугольников (левых, правых, средних)
# 2. по формуле трапеций
# 3. по формуле Симпсона


def riemann_sum_left(function, x, h):
    """Вычисление интеграла по методу левых прямоугольников"""
    return sum([h * function(x[i - 1]) for i in range(1, len(x))])


def riemann_sum_right(function, x, h):
    """Вычисление интеграла по методу правых прямоугольников"""
    return h * sum([function(x[i]) for i in range(1, len(x))])


def riemann_sum_midpoint(function, x, h):
    """Вычисление интеграла по методу средних прямоугольников"""
    return h * sum([function(x[i] - h / 2.0) for i in range(1, len(x))])


def trapezoidal_rule(function, x, h):
    """Вычисление интеграла по методу трапеций"""
    return sum([h / 2.0 * (function(x[i - 1]) + function(x[i])) for i in range(1, len(x))])


def simpsons_rule(function, x, h):
    """Вычисление интеграла по методу Симпсона"""
    return h / 3 * (function(x[0]) + sum([(2 + 2 * (i % 2)) * function(x[i]) for i in range(1, len(x))]) + function(x[-1]))


def riemann_sum_midpoint_error(second_derivative, xi, a, b, h):
    return h ** 2 * (b - a) / 24 * second_derivative(xi)


def riemann_sum_left_error(first_derivative, xi, a, b, h):
    return h * (b - a) / 2 * first_derivative(xi)


def riemann_sum_right_error(first_derivative, xi, a, b, h):
    return h * (b - a) / 2 * first_derivative(xi)


def trapezoidal_rule_error(second_derivative, xi, a, b, h):
    return - h ** 2 * (b - a) / 12 * second_derivative(xi)


def simpsons_rule_error(fourth_derivative, xi, a, b, h):
    return - h ** 4 * (b - a) / 180 * fourth_derivative(xi)


def print_integration_methods(function, derivatives, a, b, n, x, precise_value, method_names, methods, errors):
    h = (b - a) / n
    header = ['Метод', 'Точное значение', 'Прибл. значение', 'Практ. погрешность', 'Теор. погрешность']
    table = PrettyTable(header)

    for i in range(len(methods)):
        table.add_row([method_names[i],
                       precise_value,
                       methods[i](function, x, h),
                       np.absolute(precise_value - methods[i](function, x, h)),
                       errors[i](derivatives[i], b - a / 4, a, b, h)])
    print(table)


def evaluate_number_of_intervals(function, a, b, x, precise_value, method_names, methods, eps):
    header = ['Метод', 'Точное значение', 'Прибл. значение', 'Погрешность', 'Кол-во интервалов']
    table = PrettyTable(header)

    for i in range(len(methods)):
        n = 2
        h = (b - a) / n
        x, y = get_values(function, a, b, n)
        value = methods[i](function, x, h)

        while np.absolute(precise_value - value) > eps:
            n += 2
            h = (b - a) / n
            x, y = get_values(function, a, b, n)
            value = methods[i](function, x, h)

        table.add_row([method_names[i],
                       precise_value,
                       methods[i](function, x, h),
                       np.absolute(precise_value - methods[i](function, x, h)), n])
    print(table)


def main():
    a = 0
    b = np.pi / 2
    n = 6
    function = lambda u: np.cos(u)
    first_derivative = lambda u: -np.sin(u)
    second_derivative = lambda u: -np.cos(u)
    fourth_derivative = lambda u: np.cos(u)
    precise_value = 1
    x, y = get_values(function, a, b, n)
    print_values_table(x, y)

    method_names = ['Левых прямоугольников', 'Правых прямоугольников', 'Средних прямоугольников', 'Трапеций', 'Симпсона']
    methods = [riemann_sum_left, riemann_sum_right, riemann_sum_midpoint, trapezoidal_rule, simpsons_rule]
    errors = [riemann_sum_left_error, riemann_sum_right_error, riemann_sum_midpoint_error, trapezoidal_rule_error, simpsons_rule_error]
    derivatives = [first_derivative, first_derivative, second_derivative, second_derivative, fourth_derivative]

    print_integration_methods(function, derivatives, a, b, n, x, precise_value, method_names, methods, errors)
    evaluate_number_of_intervals(function, a, b, x, precise_value, method_names, methods, eps=10e-3)
    evaluate_number_of_intervals(function, a, b, x, precise_value, method_names, methods, eps=10e-5)


if __name__ == '__main__':
    main()
