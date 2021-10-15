import numpy as np

# Вариант 21
# Матрица коэффициентов системы
a = np.array([
    [4.56, 4.20, 3.78],
    [3.21, 2.73, 2.25],
    [4.58, 4.04, 3.52]
])

# Столбец свободных членов
b = np.array([
    [61.86],
    [42.98],
    [61.67]
])

# Данные для тестов. Ответ: x_1 = -0.5; x_2 = 1; x_3 = 0.5
# Матрица коэффициентов системы
a_test = np.array([
    [2., 1., 2.],
    [1., 1., 1.],
    [1., 1., 3.]
])

# Столбец свободных членов
b_test = np.array([
    [1.],
    [1.],
    [2.]
])

# Вариант 5
a_test2 = np.array([
    [1.53, 1.61, 1.43],
    [2.35, 2.31, 2.07],
    [3.83, 3.73, 3.45]
])

b_test2 = np.array([
    [-5.13],
    [-3.69],
    [-5.98]
])

tridiagonal_size = 5
m = 21
a_tridiagonal = np.zeros((tridiagonal_size, tridiagonal_size))
b_tridiagonal = np.zeros((tridiagonal_size, 1))

for i in range(tridiagonal_size):
    a_tridiagonal[i][i] = 100
    if i != tridiagonal_size - 1:
        a_tridiagonal[i][i + 1] = 0.2 * m
    if i != 0:
        a_tridiagonal[i][i - 1] = 0.3 * m

    b_tridiagonal[i][0] = m * (i + 1) * np.exp(5 / (i + 1)) * np.sin(9 / (i + 1))


# Вариант 1
a_iterative = np.array([[0.0400, -0.0029, -0.0055, -0.0082],
[0.0003, -0.05000, -0.0050, -0.0076],
[0.0008, -0.0018, -0.14000, -0.0070],
[0.0014, -0.0012, -0.0039, -0.23000]])

b_iterative = np.array(
    [[0.1220],
     [-0.2532],
     [-0.9876],
     [-2.0812]]
)


@np.vectorize
def function(x):
    """Нелинейная функция, вариант 21"""
    return np.log((1 + x) / (1 - x)) - np.cos(x) ** 2.0


@np.vectorize
def derivative(x):
    """Производная функции, вариант 21"""
    return (2.0 * x ** 2 * np.cos(x) * np.sin(x) - 2) / (x ** 2 - 1)


@np.vectorize
def iterative_function(x):
    return (np.exp(np.cos(x) ** 2) - 1) / (np.exp(np.cos(x) ** 2) + 1)


@np.vectorize
def iterative_derivative(x):
    return -(4 * np.exp(np.cos(x) ** 2) * np.cos(x) * np.sin(x)) / (np.exp(np.cos(x) ** 2) + 1) ** 2


def iterative_function_lambda(x, lambda_value):
    return x - lambda_value * (np.log((1 + x) / (1 - x)) - np.cos(x) ** 2.0)