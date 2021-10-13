import numpy as np


def bisection(start, end, k, eps, function):
    """
    Метод бисекций для нахождения корня функции f(x) на отрезке [a, b].

    :param start: начальное значение интервала a
    :param end: конечное значение интервала b
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: непрерывная функция f(x)
    :return: корень уравнения и количество пройденных итераций
    """
    mid_prev = 0.0
    for current_k in range(k):
        mid = (start + end) / 2.0

        if np.absolute(function(mid) - function(mid_prev)) < eps:
            return mid, current_k
        mid_prev = mid

        if function(mid) * function(start) < 0:
            end = mid
        elif function(mid) * function(end) < 0:
            start = mid
        elif function(mid) == 0:
            return mid, current_k
        else:
            raise Exception('Bisection method: bad initial interval.')

    return (start + end) / 2.0, k


def chord(start, end, k, eps, function):
    """
    Метод хорд для нахождения корня функции f(x) на отрезке [a, b].

    :param start: начальное значение интервала a
    :param end: конечное значение интервала b
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: непрерывная функция f(x)
    :return: корень уравнения и количество пройденных итераций
    """
    x_prev = 0.0
    for current_k in range(k):
        x = start - function(start) * (end - start) / (function(end) - function(start))

        if np.absolute(function(x) - function(x_prev)) < eps:
            return x, current_k
        x_prev = x

        if function(x) * function(start) < 0:
            end = x
        elif function(x) * function(end) < 0:
            start = x
        elif function(x) == 0 or np.absolute(function(x)) < eps:
            return x, current_k
        else:
            raise Exception('Secant method: bad initial interval.')

    return start - function(start) * (end - start) / (function(end) - function(start)), k


def newton(x, k, eps, function, derivative):
    """
    Метод Ньютона для нахождения корня функции f(x) на отрезке [a, b].

    :param x: начальное значение x
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: дифференцируемая функция f(x)
    :param derivative: производная функции f'(x)
    :return: корень уравнения и количество пройденных итераций
    """
    x_prev = 0.0
    for current_k in range(k):
        x = x - function(x) / derivative(x)

        if np.absolute(function(x) - function(x_prev)) < eps:
            return x, current_k
        x_prev = x

        if derivative(x) == 0:
            raise Exception('Newton method: zero derivative.')

    return x, k


# Doesn't work
def secant(x_prev, x, k, eps, function):
    """
    Метод секущих для нахождения корня функции f(x) на отрезке [a, b].

    :param x: начальное значение x
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: дифференцируемая функция f(x)
    :return: корень уравнения и количество пройденных итераций
    """
    for current_k in range(k):

        if np.absolute(function(x) - function(x_prev)) < eps:
            return x, current_k

        x = (x_prev * function(x) - x * function(x_prev)) / (function(x) - function(x_prev))
        x_prev = x

    return x, k
