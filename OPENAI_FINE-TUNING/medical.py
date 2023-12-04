
import pandas as pd
import tiktoken
import json

def num_tokens_from_string(String)->int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(String))
    return num_tokens

# print(num_tokens_from_string("Hello World, I'm Anantha Narayanan")) 10

def df_to_format(df):
    formatted_data=[]

    for index,row in df.iterrows():
        entry = {"message":[
            {"role":"system","content":system_prompt},
            {"role":"user","content":row['report']},
            {"role":"assistant","content":row['medical_specialty']}
        ]}
        formatted_data.append(entry)
    return formatted_data


medical_record = pd.read_csv("reports.csv")

# print(medical_record.head(5))

# print(medical_record.info())

medical_record = medical_record.dropna(subset=['report'])
# print(medical_record.info())

# print(medical_record.groupby("medical_specialty").count())

# print("Training and Validation and Testing purpose!\n")

grouped_data = medical_record.groupby('medical_specialty').sample(110,random_state=42)

# print(grouped_data['medical_specialty'].value_counts())

# print("Validation and Testing\n\n")

val_test_data = grouped_data.groupby('medical_specialty').sample(10,random_state=42)
val = val_test_data.groupby('medical_specialty').head(5)
test = val_test_data.groupby('medical_specialty').tail(5)

# print(len(val))
# print("\n\n")
# print(len(test))

train = grouped_data[~grouped_data.index.isin(val_test_data.index)]
print(len(train))

print("Number of Tokens\n")

report_lengh = train['report'].apply(num_tokens_from_string)
print(report_lengh.describe())

price_model = 0.008

print("Cost for running a model\n")
price_per_epoch = (report_lengh.sum()*price_model)/1000
print(price_per_epoch)

print(train['medical_specialty'].unique())

system_prompt = "Given medical description report, Classify it into one of these category: Cardiovascular / Pulmonary, Gastroenterology, Neurology,Radiology, Surgery"
print(system_prompt)

sample_prompt = {"message":[
    {"role":"system","content":system_prompt},
    {"role":"user","content":train['report'].iloc[0]},
    {"role":"assistant","content":train['medical_specialty'].iloc[0]}
    ]
}

print(sample_prompt)

data = df_to_format(train)

with open('fine_tuning_data.jsonl','w') as f:
    for entry in data:
        json.dump(entry,fp=f)
        # f.write(json.dump(entry))
        f.write('\n')

val_data = df_to_format(val)


with open('fine_tuning_data_val.jsonl','w') as f:
    for entry in val_data:
        json.dump(entry, fp=f)
        # f.write(json.dump(entry))
        f.write('\n')

