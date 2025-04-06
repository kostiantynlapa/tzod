import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def energy_consumption(time):
    return 2 + np.sin(2 * np.pi * time / 24) + 0.5 * np.sin(4 * np.pi * time / 24)

@njit
def monte_carlo_integral(f, a, b, n_samples):
    total = 0.0
    for _ in range(n_samples):
        x = np.random.uniform(a, b)
        total += f(x)
    return (b - a) * total / n_samples

# Тестування точності
def test_accuracy(iterations_list):
    exact_value = monte_carlo_integral(energy_consumption, 0, 24, 1000000)  # вважаємо за еталон
    results = []
    errors = []
    
    for n in iterations_list:
        estimate = monte_carlo_integral(energy_consumption, 0, 24, n)
        error = abs(estimate - exact_value)
        results.append(estimate)
        errors.append(error)
        print(f"N = {n:>7}, Estimate = {estimate:.5f}, Error = {error:.5f}")
    
    # Візуалізація похибки
    plt.figure(figsize=(10,5))
    plt.plot(iterations_list, errors, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Похибка прогнозу енергоспоживання від кількості ітерацій")
    plt.xlabel("Кількість ітерацій (лог шкала)")
    plt.ylabel("Абсолютна похибка (лог шкала)")
    plt.grid(True)
    plt.show()

iterations = [10, 100, 1000, 10000, 100000, 1000000]
test_accuracy(iterations)
