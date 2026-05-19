import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import time
import os

API_KEY = "sk-**************"
CSV_PATH = "real_data.csv"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

def check_ad_relevance(row):
    idx, data = row
    ad_id = data["Sample ID"]
    text = str(data["Text Content"])
    dialect = str(data["Dialect"])
    
    if not text.strip() or len(text.strip()) < 5:
        return {"idx": idx, "ad_id": ad_id, "dialect": dialect, "reason": "Empty Text"}
        
    prompt = f"""
    You are a professional data reviewer. Read the following advertisement text and determine if it is genuinely related to Hajj, Umrah, or Islamic travel services (agencies, visas, flights, Makkah/Madinah hotels, Ihram clothing, etc.).
    
    Text:
    "{text}"
    
    If it is deeply related to Hajj or Umrah, write exactly: YES
    If it is NOT related (e.g. cars, real estate, regular clothes, general tourism not to sacred places, electronics), write exactly: NO
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a highly precise assistant that replies with exactly one word: YES or NO."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0
            ) 
            result = response.choices[0].message.content.strip().upper()
            if "NO" in result:
                return {"idx": idx, "ad_id": ad_id, "dialect": dialect, "reason": "Not Related"}
            else:
                return None 
        except Exception as e:
            if attempt == max_retries - 1:
                return {"idx": idx, "ad_id": ad_id, "dialect": dialect, "reason": f"API Error: {e}"}
            time.sleep(2)

def main():
    print("Loading data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find file {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    total_ads = len(df)
    
    original_counts = df['Dialect'].value_counts().to_dict()
    
    print(f"Total ads loaded: {total_ads}")
    print("Pre-review Dialect Breakdown:")
    for dialect, count in original_counts.items():
        print(f"  {dialect}: {count}")
    
    print("\nStarting DeepSeek review in parallel threads...")
    
    unrelated_ads = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(check_ad_relevance, df.iterrows()), total=total_ads))
        
    for r in results:
        if r is not None:
            unrelated_ads.append(r)
            
    print("\n" + "="*50)
    print("Final Review Report")
    print("="*50)
    
    rejected_counts = {}
    for r in unrelated_ads:
        dialect = r['dialect']
        rejected_counts[dialect] = rejected_counts.get(dialect, 0) + 1
        
    total_rejected = len(unrelated_ads)
    
    print(f"Total ads reviewed: {total_ads}")
    print(f"Total ads strictly rejected (Unrelated): {total_rejected}")
    
    print("\nRejected Breakdown by Dialect:")
    for dialect in original_counts.keys():
        rej = rejected_counts.get(dialect, 0)
        print(f"  {dialect}: {rej}")
        
    print("\n==================================================")
    print(" FINAL REMAINING VALID ADS BY DIALECT ")
    print("==================================================")
    total_remaining = 0
    for dialect, orig_count in original_counts.items():
        rej = rejected_counts.get(dialect, 0)
        rem = orig_count - rej
        total_remaining += rem
        print(f"  {dialect}: {rem}")
    
    print(f"\nFinal Expected Total Valid Ads After Cleaning: {total_remaining} / 2000")
    
    if unrelated_ads:
        bad_ids = [b['ad_id'] for b in unrelated_ads]
        with open("bad_ads_ids.json", "w") as f:
            json.dump(bad_ids, f)
        print("\nSaved rejected IDs to bad_ads_ids.json")
    else:
        print("\nAll ads are valid!")

if __name__ == "__main__":
    main()
