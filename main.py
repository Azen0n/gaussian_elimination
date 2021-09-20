import numpy as np
from data import a_test2, b_test2
from gaussian_algorithm import gaussian
from norms import norm, residuals
from matrix_operations import determinant, inverse, identity

matrix = a_test2
# Размерность матрицы
n = len(a_test2[0])

x = gaussian(matrix, b_test2, out=True)
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

norm1 = norm(matrix, p=1)
norm1_inverted = norm(inverted_matrix, p=1)
normF = norm(matrix, p=2)
normF_inverted = norm(inverted_matrix, p=2)

condition_number1 = norm1 * norm1_inverted
condition_numberF = normF * normF_inverted

print('\nЧисла обусловленности:')
print('1. %.15f\n2. %.15f' % (condition_number1, condition_numberF))
