import pandas as pd
import sqlite3
import json

# -----------------------------------------------
# 1. Імітація даних із різних джерел
# -----------------------------------------------

# CSV-файл: Дані про споживання
df_csv = pd.DataFrame({
    'user_id': [1, 2, 3],
    'consumption_kwh': [120.5, 98.7, 130.2]
})
df_csv.to_csv('consumption.csv', index=False)

# JSON-файл: Дані про користувачів
json_data = [
    {"user_id": 1, "name": "Ivan", "region": "Kyiv"},
    {"user_id": 2, "name": "Olena", "region": "Lviv"},
    {"user_id": 3, "name": "Petro", "region": "Odesa"}
]
with open('users.json', 'w') as f:
    json.dump(json_data, f)

# SQL-база: Дані про погодні умови
conn = sqlite3.connect(':memory:')
df_sql = pd.DataFrame({
    'user_id': [1, 2, 3],
    'temperature': [22.1, 19.3, 25.4],
    'humidity': [45, 55, 60]
})
df_sql.to_sql('weather', conn, index=False, if_exists='replace')

# -----------------------------------------------
# 2. Завантаження з кожного джерела
# -----------------------------------------------

# CSV
consumption_df = pd.read_csv('consumption.csv')

# JSON
with open('users.json', 'r') as f:
    users_data = json.load(f)
users_df = pd.DataFrame(users_data)

# SQL
weather_df = pd.read_sql_query("SELECT * FROM weather", conn)

# -----------------------------------------------
# 3. Попередня обробка (очищення)
# -----------------------------------------------

# Перевірка на дублікати та пропуски
for df in [consumption_df, users_df, weather_df]:
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

# -----------------------------------------------
# 4. Злиття (join) у спільний датафрейм
# -----------------------------------------------

merged_df = consumption_df.merge(users_df, on='user_id')
merged_df = merged_df.merge(weather_df, on='user_id')

# -----------------------------------------------
# 5. Результат
# -----------------------------------------------

print("📊 Об’єднаний DataFrame:")
print(merged_df)
