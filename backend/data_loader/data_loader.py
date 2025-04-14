import os
import json
import pandas as pd

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "../init.json")
    
    with open(json_file_path, 'r', encoding='utf-8') as infile:
        raw_data = json.load(infile)
    
    records = raw_data if isinstance(raw_data, list) else [raw_data]
    processed_records = []
    for rec in records:
        if "ticker" in rec:
            rec['ticker'] = rec.get('ticker', 'NONE')
            rec['title'] = rec.get('title', '')
            rec['text'] = rec.get('text', '')
            rec['url'] = rec.get('url', '')
        elif "thread" in rec:
            rec['ticker'] = rec.get('ticker', 'NONE')
            rec['title'] = rec.get('thread', {}).get('title', rec.get('title', ''))
            rec['text'] = rec.get('text', '')
            rec['url'] = rec.get('thread', {}).get('url', rec.get('url', ''))
        else:
            rec['ticker'] = rec.get('ticker', 'NONE')
            rec['title'] = rec.get('title', '')
            rec['text'] = rec.get('text', '')
            rec['url'] = rec.get('url', '')
        processed_records.append(rec)
    
    df = pd.DataFrame(processed_records)
    df["combined_text"] = df["title"] + " " + df["text"]
    return df
