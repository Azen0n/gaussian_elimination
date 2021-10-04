import numpy as np
from data import a_iterative, b_iterative
from matrix_operations import convergence_rate, to_iterative_form, iterative, gaussian, calculate_error_iterative
from tabulate import tabulate


alpha, beta = to_iterative_form(a_iterative, b_iterative)
print('Исходная матрица:\n', tabulate(a_iterative, tablefmt="plain"))
print('\nМатрица, приведенная к виду x = αx + β:\n', tabulate(alpha, tablefmt="plain"))

if convergence_rate(a_iterative):
    print('\nДостаточное условие сходимости выполняется')
else:
    print('\nДостаточное условие сходимости не выполняется')

# Точность
eps = 1e-8

print('\n-----------------------Метод простых итераций-----------------------')
x = iterative(alpha, beta, eps, method='simple', out=True)
print('\n---------------------------Метод Зейделя----------------------------')
x2 = iterative(alpha, beta, eps, method='seidel', out=True)

# "Точное решение" (метод Гаусса, схема единственного деления)
x3 = gaussian(a_iterative, b_iterative, 'single')


def print_errors(x_gaussian, p_array, eps_array, method):
    x_array = []
    k_array = []
    for index, eps in enumerate(eps_array):
        x, k = iterative(alpha, beta, eps, method=method)
        x_array.append(x)
        k_array.append(k)
        print('Вектор со значениями x:\n', x_array[index])
        print('\nТочность:', eps)
        print('Количество итераций:', k_array[index])
        print('\nАбсолютная и относительная погрешности решения:')
        for p in p_array:
            absolute_error, relative_error = calculate_error_iterative(x_gaussian, x_array[index], p)
            print('При p = %s: %.15e\n\t\t\t%.15e' % (p, absolute_error, relative_error))


p_array = np.array(['1', '2', 'inf'])
eps_array = np.array([1e-8, 1e-12, 1e-15])

print('\n\n\n----------------Метод простых итераций (погрешности при разных eps)----------------')
print_errors(x3, p_array, eps_array, 'simple')

print('\n--------------------Метод Зейделя (погрешности при разных eps)---------------------')
print_errors(x3, p_array, eps_array, 'seidel')
