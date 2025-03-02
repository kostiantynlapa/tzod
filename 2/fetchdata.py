import requests
import pandas as pd

def fetch_data(dataset="AmazonScience/massive", config="af-ZA", split="train", offset=0, length=100):
  
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset, 
        "config": config, 
        "split": split, 
        "offset": offset, 
        "length": length
        }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  
        data = response.json()
        rows = data.get("rows", [])

        if not rows:
            print("API повернуло порожній список.")
            return pd.DataFrame()

        return pd.DataFrame([row["row"] for row in rows])

    except requests.RequestException as e:
        print(f"Error: {e}")
        return pd.DataFrame()
