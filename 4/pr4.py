import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pymc3 as pm

# 1. Генеруємо синтетичні дані
np.random.seed(42)
n_samples = 100
temperature = np.random.uniform(-10, 35, n_samples)
energy = 50 - 1.5 * temperature + np.random.normal(0, 3, n_samples)  # залежність зі шумом

# Готуємо датафрейм
data = pd.DataFrame({'temperature': temperature, 'energy': energy})

# 2. Класична лінійна регресія
linreg = LinearRegression()
linreg.fit(data[['temperature']], data['energy'])
classic_pred = linreg.predict(data[['temperature']])

# 3. Байєсівська регресія
with pm.Model() as model:
    # Пріори
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Лінійна модель
    mu = alpha + beta * data['temperature']

    # Ліклігуд
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data['energy'])

    # Семплінг
    trace = pm.sample(2000, tune=1000, return_inferencedata=False, cores=1, progressbar=False)

# 4. Байєсівські прогнози
post_pred = trace['alpha'][:, None] + trace['beta'][:, None] * data['temperature'].values
mean_pred = post_pred.mean(axis=0)
std_pred = post_pred.std(axis=0)

# 5. Візуалізація
plt.figure(figsize=(12, 6))
plt.scatter(data['temperature'], data['energy'], label='Дані', alpha=0.6)
plt.plot(data['temperature'], classic_pred, label='Класична лінійна регресія', color='green')
plt.plot(data['temperature'], mean_pred, label='Байєсівська регресія (середнє)', color='blue')
plt.fill_between(data['temperature'], mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                 color='blue', alpha=0.3, label='95% довірчий інтервал')
plt.xlabel("Температура (°C)")
plt.ylabel("Енергоспоживання (кВт·год)")
plt.legend()
plt.title("Порівняння класичної та байєсівської регресій")
plt.grid(True)
plt.show()
