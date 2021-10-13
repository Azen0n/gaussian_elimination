import numpy as np
import root_finding_methods as root
from data import function, derivative
from prettytable import PrettyTable


k = 30
eps = 0.00001
answer1, k1 = root.bisection(-0.4, 0.75, k, eps, function)
print(answer1)

answer2, k2 = root.chord(-0.4, 0.75, k, eps, function)
print(answer2)

answer3, k3 = root.newton(-0.4, k, eps, function, derivative)
print(answer3)

answer4, k4 = root.secant(0.1, 0.9, k, eps, function)
print(answer4)

table = PrettyTable(['Метод', 'Начальное приближение', 'Полученный корень', 'Число итераций', 'Величина погрешности ε', 'Проверка f(ξ)'])
table.add_row(['Метод бисекций',    [-0.4, 0.75],   answer1, k1, eps, np.round(function(answer1), 2) == 0.0])
table.add_row(['Метод хорд',        [-0.4, 0.75],   answer2, k2, eps, np.round(function(answer2), 2) == 0.0])
table.add_row(['Метод Ньютона',     -0.4,           answer3, k3, eps, np.round(function(answer3), 2) == 0.0])
table.add_row(['Метод секущих',     0.9,            answer4, k4, eps, np.round(function(answer4), 2) == 0.0])
print(table)
