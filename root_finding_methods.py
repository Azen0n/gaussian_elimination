import numpy as np
from data import iterative_function


def bisection(start, end, k, eps, function):
    """
    Метод бисекций для нахождения корня функции f(x) на отрезке [a, b].

    :param start: начальное значение интервала a
    :param end: конечное значение интервала b
    :param k: количество итераций
    :param eps: значение точности
    :param function: непрерывная функция f(x)
    :return: корень уравнения, количество пройденных итераций и список со всеми найденными значениями x
    """
    mid_prev = 0.0
    plot_data = []
    for current_k in range(k):
        mid = (start + end) / 2.0

        if np.absolute(function(mid) - function(mid_prev)) < eps:
            return mid, current_k, plot_data
        mid_prev = mid

        plot_data.append(mid)

        if function(mid) * function(start) < 0:
            end = mid
        elif function(mid) * function(end) < 0:
            start = mid
        elif function(mid) == 0:
            return mid, current_k, plot_data
        else:
            raise Exception('Bisection method: bad initial interval.')

    return (start + end) / 2.0, k, plot_data


def chord(start, end, k, eps, function):
    """
    Метод хорд для нахождения корня функции f(x) на отрезке [a, b].

    :param start: начальное значение интервала a
    :param end: конечное значение интервала b
    :param k: количество итераций
    :param eps: значение точности
    :param function: непрерывная функция f(x)
    :return: корень уравнения, количество пройденных итераций и список со всеми найденными значениями x
    """
    x_prev = 0.0
    plot_data = []
    for current_k in range(k):
        x = start - function(start) * (end - start) / (function(end) - function(start))
        plot_data.append(x)

        if np.absolute(function(x) - function(x_prev)) < eps:
            return x, current_k, plot_data
        x_prev = x

        if function(x) * function(start) < 0:
            end = x
        elif function(x) * function(end) < 0:
            start = x
        elif function(x) == 0 or np.absolute(function(x)) < eps:
            return x, current_k, plot_data
        else:
            raise Exception('Secant method: bad initial interval.')

    return start - function(start) * (end - start) / (function(end) - function(start)), k, plot_data


def newton(x, k, eps, function, derivative):
    """
    Метод Ньютона для нахождения корня функции f(x).

    :param x: начальное приближение x
    :param k: количество итераций
    :param eps: значение точности
    :param function: дифференцируемая функция f(x)
    :param derivative: производная функции f'(x)
    :return: корень уравнения, количество пройденных итераций и список со всеми найденными значениями x
    """
    x_prev = 0.0
    plot_data = []
    for current_k in range(k):
        plot_data.append(x)

        x = x - function(x) / derivative(x)

        if np.absolute(function(x) - function(x_prev)) < eps:
            return x, current_k, plot_data
        x_prev = x

        if derivative(x) == 0:
            raise Exception('Newton method: zero derivative.')

    return x, k, plot_data


def secant(x_prev, x, k, eps, function):
    """
    Метод секущих для нахождения корня функции f(x).

    :param x_prev: первое начальное приближение x
    :param x: второе начальное приближение x
    :param k: количество итераций
    :param eps: значение точности
    :param function: дифференцируемая функция f(x)
    :return: корень уравнения, количество пройденных итераций и список со всеми найденными значениями x
    """
    plot_data = [x_prev]
    for current_k in range(k):
        plot_data.append(x)

        if np.absolute(function(x) - function(x_prev)) < eps:
            return x, current_k, plot_data

        x, x_prev = x - function(x) * (x - x_prev) / (function(x) - function(x_prev)), x

    return x, k, plot_data


def iterative(x, k, eps, function, lambda_value=None):
    """
    Метод простых итераций для нахождения корня функции f(x).

    :param x: начальное приближение x
    :param k: количество итераций
    :param eps: значение точности
    :param function: дифференцируемая функция f(x)
    :param lambda_value: значение λ (по умолчанию None)
    :return: корень уравнения, количество пройденных итераций и список со всеми найденными значениями x
    """
    plot_data = []
    for current_k in range(k):
        plot_data.append(x)

        x_prev = x
        x = iterative_function(x)

        if lambda_value is not None:
            if np.absolute(function(x, lambda_value) - function(x_prev, lambda_value)) < eps:
                return x, current_k, plot_data
        else:
            if np.absolute(function(x) - function(x_prev)) < eps:
                return x, current_k, plot_data

    return x, k, plot_data
