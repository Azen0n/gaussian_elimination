import numpy as np
import root_finding_methods as root
import data
from prettytable import PrettyTable
from matplotlib import pyplot as plt


# График function
x = np.array(np.arange(-0.99, 0.99, 0.05))
y = data.function(x)

plt.plot(x, y)
y1 = np.zeros(x.shape)
plt.plot(x, y1, c='red')

plt.plot(0.4, 0, 'ro', ms=4)
plt.plot([0.4, 0.4], [0, -5.59], '--', c='gray')

plt.text(0.15, 2.0, r'$f(x)=ln\frac{1 + x}{1 - x}-cos^2(x)$')
plt.text(-0.49, 0.25, r'$f(x)=0$')
plt.text(0.3, 0.25, r'$ξ$')

plt.title('Графическое отделение корня уравнения')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()


k = 30
eps = 1e-06
a = 0.25        # Интервал [a; b]
b = 0.5
root_bisection, k_bisection, plot_data_bisection = root.bisection(a, b, k, eps, data.function)
root_chord, k_chord, plot_data_chord = root.chord(a, b, k, eps, data.function)
root_newton, k_newton, plot_data_newton = root.newton(a, k, eps, data.function, data.derivative)
root_secant, k_secant, plot_data_secant = root.secant(a, a + 0.1, k, eps, data.function)


table = PrettyTable(['Метод', 'Начальное приближение', 'Полученный корень',
                     'Число итераций', 'Величина погрешности ε', 'Проверка f(ξ)'])
table.add_row(['Метод бисекций', plot_data_bisection[0], root_bisection, k_bisection, eps, np.round(data.function(root_bisection), 2) == 0.0])
table.add_row(['Метод хорд', plot_data_chord[0], root_chord, k_chord, eps, np.round(data.function(root_chord), 2) == 0.0])
table.add_row(['Метод Ньютона', plot_data_newton[0], root_newton, k_newton, eps, np.round(data.function(root_newton), 2) == 0.0])
table.add_row(['Метод секущих', plot_data_secant[0], root_secant, k_secant, eps, np.round(data.function(root_secant), 2) == 0.0])
print(table)


def plot(plot_data, k, title):
    plt.plot(data.function(plot_data))
    plt.title(title)
    plt.xlabel('Количество итераций')
    plt.ylabel('f(x)')
    plt.xticks(range(0, k + 1))
    plt.grid(axis='y')
    plt.show()


# Графики изменения f(x) четырех методов
plot(plot_data_bisection, k_bisection, 'Метод бисекций')
plot(plot_data_chord, k_chord, 'Метод хорд')
plot(plot_data_newton, k_newton, 'Метод Ньютона')
plot(plot_data_secant, k_secant, 'Метод секущих')


# Метод простых итераций: таблица с результатами для разных точностей и график изменения f(x)
def print_iterative_results(function, lambda_value=None):
    table = PrettyTable(['Начальное приближение', 'Полученный корень',
                                   'Число итераций', 'Величина погрешности ε', 'Проверка f(ξ)'])
    for eps_iterative in [1e-03, 1e-05, 1e-08]:
        if lambda_value is not None:
            root_iterative, k_iterative, plot_data_iterative = root.iterative(a, k, eps_iterative, function, lambda_value)
            table.add_row([plot_data_iterative[0], root_iterative, k_iterative, eps_iterative,
                                     np.round(data.function(root_iterative), 2) == 0.0])
        else:
            root_iterative, k_iterative, plot_data_iterative = root.iterative(a, k, eps_iterative, function)
            table.add_row([plot_data_iterative[0], root_iterative, k_iterative, eps_iterative,
                                     np.round(function(root_iterative), 2) == 0.0])
    print(table)
    plot(plot_data_iterative, k_iterative, 'Метод простых итераций')


print('\nМетод простых итераций')
derivative_value = data.iterative_derivative(root_bisection)
print(f'Проверка условия сходимости: |f\'(ξ)| = {np.absolute(derivative_value):.2f} '
      f'{"<= q < 1" if derivative_value < 1 else "> 1 — условие сходимости не выполняется"}')
print_iterative_results(data.function)


# Расчеты для двух наборов значений λ и q
alpha_iterative = -0.15
gamma_iterative = -0.35

lambda_monotone = 1.0 / gamma_iterative
q_monotone = 1.0 - alpha_iterative / gamma_iterative

print(f'\nМетод простых итераций при λ = {lambda_monotone} и q = {q_monotone} (монотонная сходимость)')
print(f'Проверка условия сходимости: |f\'(ξ)| = {np.absolute(derivative_value):.2f} '
      f'{"<= q < 1" if np.absolute(derivative_value) <= q_monotone else ">= q — условие сходимости не выполняется"}')
print_iterative_results(data.iterative_function_lambda, lambda_monotone)

lambda_optimal = 2.0 / (alpha_iterative + gamma_iterative)
q_optimal = (gamma_iterative - alpha_iterative) / (gamma_iterative + alpha_iterative)

print(f'\nМетод простых итераций при λ = {lambda_optimal} и q = {q_optimal} (оптимальная сходимость)')
print(f'Проверка условия сходимости: |f\'(ξ)| = {np.absolute(derivative_value):.2f} '
      f'{"<= q < 1" if np.absolute(derivative_value) <= q_optimal else ">= q — условие сходимости не выполняется"}')
print_iterative_results(data.iterative_function_lambda, lambda_optimal)
