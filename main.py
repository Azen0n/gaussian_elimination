import numpy as np
from data import a_test2, b_test2
from matrix_operations import gaussian, condition_number, determinant, inverse, identity, constant_terms_error, \
    coefficients_error
from norms import residuals

matrix = a_test2
# Размерность матрицы
n = len(a_test2[0])

x = gaussian(matrix, b_test2, out=True, method='single')
x2 = gaussian(matrix, b_test2, out=True, method='pivot')
residuals(matrix, b_test2, x)
print('\nОпределитель матрицы: ', determinant(matrix))
print('\nОбратная матрица:\n', inverse(matrix))

inverted_matrix = inverse(matrix)
product = np.absolute(matrix.dot(inverted_matrix).round())
identity_matrix = identity(n)
print('\nРезультат умножения матрицы на ее обратную:\n', product)
if np.array_equal(product, identity_matrix):
    print('Получена единичная матрица. Равенство верно.')
else:
    print('Получена не единичная матрица. Равенство неверно.')

condition_number1, condition_number2 = condition_number(matrix)
print('\nЧисла обусловленности:')
print('1. %.15f\n2. %.15f\n' % (condition_number1, condition_number2))

constant_terms_error(a_test2, b_test2, 0.01, method='single')
coefficients_error(a_test2, b_test2, 0.01, method='single')
