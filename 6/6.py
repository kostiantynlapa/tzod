import pandas as pd
import dask.dataframe as dd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def generate_big_data(n_rows=1_500_000):
    """Generate synthetic dataset with numerical and categorical columns."""
    np.random.seed(42)
    data = {
        'id': np.arange(n_rows),
        'value1': np.random.normal(loc=0, scale=1, size=n_rows),
        'value2': np.random.randint(0, 100, size=n_rows),
        'category': np.random.choice(['A', 'B', 'C'], size=n_rows)
    }
    return pd.DataFrame(data)

def plot_distributions(df):
    """Visualize distributions of numerical and categorical features."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df['value1'], bins=50, ax=axes[0], kde=True, color="skyblue")
    axes[0].set_title("Distribution of value1")

    sns.histplot(df['value2'], bins=50, ax=axes[1], color="salmon")
    axes[1].set_title("Distribution of value2")

    sns.countplot(x='category', data=df, ax=axes[2], palette="Set2")
    axes[2].set_title("Category counts")

    plt.tight_layout()
    plt.show()

def process_with_pandas(df):
    """Run basic statistics and group operations using pandas."""
    start = time.time()
    stats = df.describe()
    filtered = df[df['value2'] > 50]
    grouped = df.groupby('category')['value1'].mean()
    duration = time.time() - start
    print(f"Pandas обробка завершена за {duration:.2f} секунд.")
    return stats, filtered, grouped, duration

def process_with_dask(df):
    """Run basic statistics and group operations using dask."""
    ddf = dd.from_pandas(df, npartitions=8)
    start = time.time()
    stats = ddf.describe().compute()
    filtered = ddf[ddf['value2'] > 50].compute()
    grouped = ddf.groupby('category')['value1'].mean().compute()
    duration = time.time() - start
    print(f"Dask обробка завершена за {duration:.2f} секунд.")
    return stats, filtered, grouped, duration

def plot_group_comparison(pandas_grouped, dask_grouped):
    """Compare mean value1 between Pandas and Dask."""
    comparison_df = pd.DataFrame({
        'Pandas': pandas_grouped,
        'Dask': dask_grouped
    })

    comparison_df.plot(kind='bar', figsize=(8, 5), color=['#5DADE2', '#48C9B0'])
    plt.title("Comparison of Mean value1 by Category")
    plt.ylabel("Mean value1")
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare():
    df = generate_big_data()
    print("Згенеровано набір даних обсягом:", df.shape)

    plot_distributions(df)

    pandas_stats, pandas_filtered, pandas_grouped, pandas_time = process_with_pandas(df)
    dask_stats, dask_filtered, dask_grouped, dask_time = process_with_dask(df)

    print("\n--- Порівняння часу виконання ---")
    print(f"Pandas: {pandas_time:.2f} с")
    print(f"Dask:   {dask_time:.2f} с")

    print("\n--- Середні значення value1 за категоріями (Pandas) ---")
    print(pandas_grouped)
    print("\n--- Середні значення value1 за категоріями (Dask) ---")
    print(dask_grouped)

    plot_group_comparison(pandas_grouped, dask_grouped)

if __name__ == "__main__":
    compare()
