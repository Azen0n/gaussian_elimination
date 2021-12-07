import numpy as np
from prettytable import PrettyTable
import scipy.integrate as integrate

from polynomial import print_values_table


# Вычислить интеграл по формулам:
# 1. прямоугольников (левых, правых, средних)
# 2. по формуле трапеций
# 3. по формуле Симпсона


def riemann_sum_left(function, x, h):
    """Вычисление интеграла по методу левых прямоугольников"""
    return h * sum([function(x[i - 1]) for i in range(1, len(x))])


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
    return h / 3 * sum([(2 + 2 * (i % 2)) * function(x[i]) for i in range(len(x))])


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
        n = 4
        h = (b - a) / n
        value = methods[i](function, x, h)

        while np.absolute(precise_value - value) > eps:
            x, y = print_values_table(function, a, b, n)
            n += 2
            h = (b - a) / n
            value = methods[i](function, x, h)

        table.add_row([method_names[i],
                       precise_value,
                       methods[i](function, x, h),
                       np.absolute(precise_value - methods[i](function, x, h)), n])
    print(table)


def main():
    a = 0
    b = 2 * np.pi
    n = 6
    function = np.sin
    first_derivative = np.cos
    second_derivative = lambda u: -np.sin(u)
    fourth_derivative = np.sin
    precise_value = integrate.quad(function, a, b)[0]
    x, y = print_values_table(function, a, b, n)

    method_names = ['Левых прямоугольников', 'Правых прямоугольников', 'Средних прямоугольников', 'Трапеций', 'Симпсона']
    methods = [riemann_sum_left, riemann_sum_right, riemann_sum_midpoint, trapezoidal_rule, simpsons_rule]
    errors = [riemann_sum_left_error, riemann_sum_right_error, riemann_sum_midpoint_error, trapezoidal_rule_error, simpsons_rule_error]
    derivatives = [first_derivative, first_derivative, second_derivative, second_derivative, fourth_derivative]

    print_integration_methods(function, derivatives, a, b, n, x, precise_value, method_names, methods, errors)
    evaluate_number_of_intervals(function, a, b, x, precise_value, method_names, methods, eps=10e-5)
    evaluate_number_of_intervals(function, a, b, x, precise_value, method_names, methods, eps=10e-6)
    evaluate_number_of_intervals(function, a, b, x, precise_value, method_names, methods, eps=10e-7)


if __name__ == '__main__':
    main()
