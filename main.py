import numpy as np
from data import a_test2, b_test2
from gaussian_algorithm import gaussian
from norms import residuals
from matrix_operations import determinant

x = gaussian(a_test2, b_test2, out=True)
residuals(a_test2, b_test2, x)
print('\nОпределитель матрицы: ', determinant(a_test2))
