import numpy as np
from data import a, b, a_test2, b_test2, a_tridiagonal, b_tridiagonal
from matrix_operations import gaussian, condition_number, determinant, inverse, identity, constant_terms_error, \
    coefficients_error
from norms import residuals
np.set_printoptions(precision=15)

answer = ''
while answer not in ['1', '2', '3', '4']:
    print('\nВыберите источник данных:'
          '\n 1. Вариант 21'
          '\n 2. Вариант 5'
          '\n 3. Случайная генерация'
          '\n 4. Ввод с клавиатуры\n')
    answer = input()

if answer == '1':
    matrix = a              # Матрица A
    constant_terms = b      # Столбец свободных членов b
    n = len(matrix[0])      # Размерность матрицы

elif answer == '2':
    matrix = a_test2
    constant_terms = b_test2
    n = len(matrix[0])      # Размерность матрицы

elif answer == '3':
    print('Введите размерность матрицы: ')
    n = int(input())
    matrix = np.random.uniform(1.0, 10.0, (n, n))
    constant_terms = np.random.uniform(10.0, 20.0, (n, 1))

elif answer == '4':
    print('Введите размерность матрицы:', end=' ')
    n = int(input())

    matrix = np.zeros((n, n))
    constant_terms = np.zeros((n, 1))
    print('Введите элементы матрицы A, отделяя цифры точкой:')
    for i in range(n):
        for j in range(n):
            print('[%s][%s]:' % (i, j), end=' ')
            element = float(input())
            matrix[i][j] = element

    print('Введите элементы столбца свободных членов b, отделяя цифры точкой:')
    for i in range(n):
        print('[%s]:' % i, end=' ')
        element = float(input())
        constant_terms[i][0] = element

print('\n--------------Метод Гаусса: схема единственного деления--------------')
x = gaussian(matrix, constant_terms, out=True, method='single')
print('\n--------------Метод Гаусса: с выбором главного элемента--------------')
x2 = gaussian(matrix, constant_terms, out=True, method='pivot')
print('\n----------------Метод Гаусса: LU-разложение матрицы A----------------')
x3 = gaussian(matrix, constant_terms, out=True, method='lu')

residuals(matrix, constant_terms, x)
print('\nОпределитель матрицы: ', determinant(matrix))
print('\nОбратная матрица:\n', inverse(matrix))

inverted_matrix = inverse(matrix)
product = np.absolute(matrix.dot(inverted_matrix))
identity_matrix = identity(n)
print('\nРезультат умножения матрицы на ее обратную:\n', product)
if np.array_equal(product.round(), identity_matrix):
    print('Получена единичная матрица. Равенство верно.')
else:
    print('Получена не единичная матрица. Равенство неверно.')

condition_number1, condition_number2 = condition_number(matrix)
print('\nЧисла обусловленности:')
print('1. %.15f\n2. %.15f\n' % (condition_number1, condition_number2))

constant_terms_error(matrix, constant_terms, 0.01, method='single')
coefficients_error(matrix, constant_terms, 0.01, method='single')

print('\n------------Метод квадратных корней (разложение Холецкого)------------')
x4 = gaussian(matrix.T.dot(matrix), constant_terms, out=True, method='cholesky')
print('\n----------------------------Метод прогонки----------------------------')
x5 = gaussian(a_tridiagonal, b_tridiagonal, out=True, method='tridiagonal')
