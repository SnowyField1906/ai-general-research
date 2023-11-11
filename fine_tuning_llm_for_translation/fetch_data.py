import os
import csv
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

absolute_path = os.path.dirname(os.path.abspath(__file__))
file_path = absolute_path + '/data.csv'

def parse_json():
    output_file_path = absolute_path + '/output.json'

    with open(file_path, newline='', encoding='UTF-8') as f, open(output_file_path, 'w', encoding='utf-8') as output_file:
        reader = csv.reader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        next(reader, None)

        json_data = []

        for row in reader:
            cleaned_row = [cell.strip().encode('utf-8').decode('utf-8') for cell in row if cell.strip()]

            json_object = {
                "prompt": cleaned_row[0],
                "response": cleaned_row[1]
            }

            json_data.append(json_object)
        
        json.dump(json_data, output_file, ensure_ascii=False, indent=2)

def parse_jsonl():
    output_file_path = absolute_path + '/output.jsonl'
    role = "You are a native Japanese speaker. Your job is to send the refined sentence to correct the Japanese sentence that user submit to look natural and accurate."

    with open(file_path, newline='', encoding='UTF-8') as f, open(output_file_path, 'w', encoding='utf-8') as output_file:
        reader = csv.reader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        next(reader, None)

        for row in reader:
            cleaned_row = [cell.strip().encode('utf-8').decode('utf-8') for cell in row if cell.strip()]

            json_object = {
                "messages": [
                    {"role": "system", "content": role},
                    {"role": "user", "content": cleaned_row[0].replace('\r\n', '\n')},
                    {"role": "assistant", "content": cleaned_row[1].replace('\r\n', '\n')}
                ]
            }

            # Convert the JSON object to a single-line string and write it as a line
            output_file.write(json.dumps(json_object, ensure_ascii=False, separators=(',', ':')) + '\n')

def parse_parquet():
    json_file_path = absolute_path + '/output.json'
    output_file_path = absolute_path + '/output.parquet'

    data = json.load(open(json_file_path, 'r', encoding='utf-8'))
    df = pd.DataFrame(data)

    # Convert DataFrame to Arrow Table
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file_path)


parse_jsonl()