
import os
import openai
import json
from constant import anantha_api, kamalesh_oepnai
# os.environ['OPENAI_API_KEY']=anantha_api
from openai import OpenAI
from medical import test,system_prompt

openai.api_key=anantha_api

client = OpenAI()

file_upload_response = client.files.create(
    file=open(r'fine_tuning_data.jsonl','rb'),
    purpose='fine-tune'
)
print(file_upload_response) #check status->processed

file_upload_response_val= client.files.create(
    file=open(r"fine_tuning_data_val.jsonl","rb"),
    purpose="fine-tune"
)

print(file_upload_response_val)

fine_tuninig_response = client.fine_tuning.jobs.create(
    training_file=file_upload_response.id,
    model="gpt-3.5-turbo",
    hyperparameters={'n_epochs':1},
    validation_file=file_upload_response_val.id
)

print(fine_tuninig_response)

print(client.fine_tuning.jobs.list(limit=1)) #check for finished_at & status (Takes time)

'''
You can see the Training loss and validation loss in OpenAI platform
'''

training_event = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id="ftjob_x5VM5PdPJ5MHgVZxS3tlrsFD",
    limit=500, #no of epochs
)

print(training_event.data) #fine tuning job events

print(training_event.data[2].data) # return dictionary

train_loss = []
val_loss = []

for item in training_event.data:
    train_data = item.data
    if train_data and 'train_loss' in train_data:
        train_loss.insert(0,train_data['train_loss'])
        val_loss.insert(0,train_data['valid_loss'])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)

'''
Pass the model to chat completion API
'''

test_report = test['report'].iloc[1]

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {'role':'system','content':system_prompt},
        {'role':'user','content':test_report}
    ]
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="your-model-name-ID",
    messages=[
        {'role':'system','content':system_prompt},
        {'role':'user','content':test_report}
    ]
)
print(completion.choices[0].message.content.strip())

def classify_report(report,model):
    completion1 = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content':report}
        ]
    )

    return completion1

predited_classes = []
ground_truth_classes = []

for line in test.iterrows():
    report,specialty = line[1]['report'],line[1]['medical_specialty']
    ground_truth_classes.append(specialty.strip())
    prediction = classify_report(report=report,model='gpt-3.5-turbo')
    predited_classes.append(prediction.choices[0].message.content.strip())

import numpy as np

print((np.array(predited_classes)==np.array(ground_truth_classes)).mean()) #Accuracy 0.52

for line in test.iterrows():
    report,specialty = line[1]['report'],line[1]['medical_specialty']
    ground_truth_classes.append(specialty.strip())
    prediction = classify_report(report=report,model='your-model-name-ID')
    predited_classes.append(prediction.choices[0].message.content.strip())

print((np.array(predited_classes) == np.array(ground_truth_classes)).mean())  # Accuracy 0.6

'''
Next to improve the perforamncewould be to enlarge our training dataset & epochs
'''