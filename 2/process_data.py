import pandas as pd
from fetchdata import fetch_data

df = fetch_data(config="af-ZA", length=100)

if df.empty:
    print("DataFrame порожній")
    exit()


def extract_scores(judgments):
    if isinstance(judgments, dict) and "intent_score" in judgments:
        return {
            "intent_avg": sum(judgments["intent_score"]) / len(judgments["intent_score"]),
            "slots_avg": sum(judgments["slots_score"]) / len(judgments["slots_score"]),
            "grammar_avg": sum(judgments["grammar_score"]) / len(judgments["grammar_score"]),
            "spelling_avg": sum(judgments["spelling_score"]) / len(judgments["spelling_score"]),
        }
    return {"intent_avg": None, "slots_avg": None, "grammar_avg": None, "spelling_avg": None}


score_df = df["judgments"].apply(extract_scores).apply(pd.Series)
df = pd.concat([df, score_df], axis=1).drop(columns=["judgments"])

if {"intent_avg", "slots_avg", "grammar_avg", "spelling_avg"}.issubset(df.columns):
    grouped = df.groupby("locale")[["intent_avg", "slots_avg", "grammar_avg", "spelling_avg"]].mean()
    print("\nСередні оцінки за мовами:")
    print(grouped)
else:
    print("Error")

print("\nПерші 5 рядків DataFrame:")
print(df.head())
