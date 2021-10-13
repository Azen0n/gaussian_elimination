import numpy as np


def bisection(start, end, k, eps, function):
    """
    Метод бисекций для нахождения корня функции f(x) на отрезке [a, b].

    :param start: начальное значение интервала a
    :param end: конечное значение интервала b
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: непрерывная функция f(x)
    :return: корень уравнения
    """
    for _ in range(k):
        mid = (start + end) / 2.0

        if function(mid) * function(start) < 0:
            end = mid
        elif function(mid) * function(end) < 0:
            start = mid
        elif function(mid) == 0 or np.absolute(function(mid)) < eps:
            return mid
        else:
            raise Exception('Bisection method: bad initial interval.')

    return (start + end) / 2.0


def chord(start, end, k, eps, function):
    """
    Метод хорд для нахождения корня функции f(x) на отрезке [a, b].

    :param start: начальное значение интервала a
    :param end: конечное значение интервала b
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: непрерывная функция f(x)
    :return: корень уравнения
    """
    for _ in range(k):
        x = start - function(start) * (end - start) / (function(end) - function(start))

        if function(x) * function(start) < 0:
            end = x
        elif function(x) * function(end) < 0:
            start = x
        elif function(x) == 0 or np.absolute(function(x)) < eps:
            return x
        else:
            raise Exception('Secant method: bad initial interval.')

    return start - function(start) * (end - start) / (function(end) - function(start))


def newton(x, k, eps, function, derivative):
    """
    Метод Ньютона для нахождения корня функции f(x) на отрезке [a, b].

    :param x: начальное значение x
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: дифференцируемая функция f(x)
    :param derivative: производная функции f'(x)
    :return: корень уравнения
    """
    for _ in range(k):
        x = x - function(x) / derivative(x)
        if np.absolute(function(x)) < eps:
            return x
        if derivative(x) == 0:
            raise Exception('Newton method: zero derivative.')

    return x


def secant(x, k, eps, function):
    """
    Метод секущих для нахождения корня функции f(x) на отрезке [a, b].

    :param x: начальное значение x
    :param k: количество итераций
    :param eps: значение погрешности
    :param function: дифференцируемая функция f(x)
    :return: корень уравнения
    """
    for _ in range(k):
        if np.absolute(function(x)) < eps:
            return x
        x_prev = x
        x = x - (function(x) * (x - x_prev)) / (function(x) * function(x_prev))

    return x
