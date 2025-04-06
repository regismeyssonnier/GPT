# -*- coding: latin -*-

import json
import random

data = []

train_data_subjects = ['regis']
train_data_verbs = ['is', 'seems', 'looks', 'appears to be']
train_data_prn = ['the most', 'very', 'extremely', 'super', 'incredibly']
train_data_nom = ['beautiful', 'strong', 'clever', 'cute', 'tall', 'brilliant', 'kind', 'fast', 'wise', 'funny']
train_data_templates = [
    "{subj} {verb} {prn} {adj}",
    "{subj} {verb} really {adj}",
    "{subj} is truly {adj}",
    "without a doubt, {subj} is {prn} {adj}",
    "everyone knows {subj} is {prn} {adj}",
]

for i in range(200):
    template = random.choice(train_data_templates)
    sentence = template.format(
        subj=random.choice(train_data_subjects),
        verb=random.choice(train_data_verbs),
        prn=random.choice(train_data_prn),
        adj=random.choice(train_data_nom)
    )
    data.append({"text": sentence})

# Sauvegarde dans un fichier JSON
file_name = "regis.json"
with open(file_name, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Les donn?es enrichies ont ?t? sauvegard?es dans {file_name}")
