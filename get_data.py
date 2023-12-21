#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   get_data.py
@Time    :   2023/01/16 13:47:46
@Author  :   Bo 
'''


import pandas as pd

def get_wind_train_dataset(conf):
    local_id = conf.use_local_id
    train_dataset_client = []

    # Read in subset based on local_id
    with open(f'clients_single_blade/client_{local_id}.json', 'r') as f:
            train_dataset_client = pd.read_json(f, orient='records', lines=True)

    # Convert JSON string to DataFrame
    return train_dataset_client

def get_wind_test_dataset():
    train_dataset_client = []

    # Read in subset based on local_id
    with open(f'clients_single_blade/test_data.json', 'r') as f:
            train_dataset_client = pd.read_json(f, orient='records', lines=True)

    return train_dataset_client
    
