import json
import os
import pickle
import sys
import csv
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from time import time
from data_loader.dataset import DataSet
from modules.model import GGNAT
from trainer import train
from utils import tally_param, debug

from gensim.models import word2vec, Doc2Vec

header = ['data_name', 'acc', 'pr', 'rc', 'f1', 'window_size']


torch.manual_seed(1000)
np.random.seed(1000)
torch.cuda.empty_cache()



def run(data_dir, data_name, window_size, model_name = "GGNAT", pool_layer=2, num_heads=3, epoch=100):
    model_dir = os.path.join('models', 'statement')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = "dataset"
    processed_data_path = os.path.join(input_dir, data_name+"_"+window_size+'_'+'statement_processed.bin')
    if os.path.exists(processed_data_path) :
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        train_data_path, test_data_path = load_data(data_dir, data_name)
        dataset = DataSet(train_src=train_data_path,                        
                          valid_src=None,
                          test_src=test_data_path,
                          batch_size=64, n_ident='node_features', g_ident='graph',
                          l_ident='label', window_size=window_size, is_token = False)
        
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
    
    model = GGNAT(input_dim=dataset.feature_size, output_dim=dataset.feature_size,
                            num_steps=6, num_heads=num_heads, max_edge_types=dataset.max_edge_type, pool_layer=pool_layer)

    model_path = os.path.join(model_dir, data_name+'-model.bin')
    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    if True:#not os.path.exists(model_path):
        dev_every = int(len(dataset.train_examples)/ dataset.batch_size) + 1
        max_steps = dev_every * epoch
        train(model=model, dataset=dataset, max_steps=max_steps, dev_every=dev_every,
            loss_function=loss_function, optimizer=optim,
            save_path=model_path, max_patience=100, log_every=None)

    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    from trainer import evaluate_metrics
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)                                
    debug('\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (acc, pr, rc, f1))
    metric= {
        'data_name': data_name,
        'acc': round(acc,3),
        'pr': round(pr,3),
        'rc': round(rc,3),
        'f1': round(f1,3),
        'window_size': window_size,
    }
    return metric


if __name__ == '__main__':

  
    for data_name in os.listdir("dataset\\CWE"):          
        if data_name.endswith('txt'):
            with open('result\\gcn2at\\'+data_name.split('.')[0]+'.csv', 'a', encoding='utf-8') as f:
                csvfile = csv.DictWriter(f, fieldnames=header)
                csvfile.writeheader()
                for window in ['2', '3', '4', '5', '6']:
                    metric = run("dataset\\CWE", data_name, window_size=window, pool_layer=2, num_heads=3, epoch=100)
                    csvfile.writerow(metric)      
         