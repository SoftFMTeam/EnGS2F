import copy
import json
from random import random
import sys
import numpy as np
import torch
from dgl import DGLGraph
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug
from data_loader.data_process import *
import torch_geometric.transforms as T

from torch_geometric.utils import degree
from torch_geometric.data import Dataset, Data, Batch
from random import choice
def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]



class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target, window_size):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)

        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges[window_size]:
            if type(s) == str:
                continue
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})
            

class DataSet:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32, n_ident=None, g_ident=None, l_ident=None, window_size=2,is_token = False):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.window_size = window_size
        self.is_token = is_token
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(test_src, train_src, valid_src)
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, test_src, train_src, valid_src):
        debug('Reading Train File!')
        if train_src is not None:
            with open(train_src) as fp:
                train_data = json.load(fp)
                for entry in tqdm(train_data):             
                    if not self.is_token:  
                        ast_features = entry[self.n_ident]             
                        dv_features = entry['dv_node_features']      
                        node_features = [dv_features[i] + ast_features[i] for i in range(len(entry[self.n_ident]))]
                        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features= node_features,
                                            edges=entry[self.g_ident], target=entry['label'],window_size=self.window_size)
                    else:
                        example = DataEntry(datset=self, num_nodes=len(entry['token_node_features']), features=entry['token_node_features'],
                                        edges=entry['token_graph'], target=entry['label'],window_size=self.window_size)   
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    self.train_examples.append(example)
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                valid_data = json.load(fp)
                for entry in tqdm(valid_data):
                    graph = entry['graph']
                    example = DataEntry(datset=self, num_nodes=len(graph[self.n_ident]),
                                        features=graph[self.n_ident],
                                        edges=graph[self.g_ident], target=entry['label'],window_size=self.window_size)
                    self.valid_examples.append(example)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = json.load(fp)
                for entry in tqdm(test_data):                             
                    if not self.is_token:        
                        ast_features = entry[self.n_ident]             
                        dv_features = entry['dv_node_features']      
                        node_features = [dv_features[i] + ast_features[i] for i in range(len(entry[self.n_ident]))]    
                        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features= node_features,
                                            edges=entry[self.g_ident], target=entry['label'],window_size=self.window_size)
                    else:
                        example = DataEntry(datset=self, num_nodes=len(entry['token_node_features']), features=entry['token_node_features'],
                                        edges=entry['token_graph'], target=entry['label'],window_size=self.window_size)
                    self.test_examples.append(example)

    def read_dataset2(self, test_src, train_src, valid_src):
        debug('Reading Train File!')
        if train_src is not None:
            with open(train_src) as fp:
                for line in fp.readlines():
                    line = line.strip('\n')
                    graph = json.loads(line)
                  
                    example = DataEntry(datset=self, num_nodes=len(graph[self.n_ident]),
                                        features=graph[self.n_ident],
                                        edges=graph[self.g_ident], target=graph['target'][0])
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    self.train_examples.append(example)
       
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                for line in fp.readlines():
                    line = line.strip('\n')
                    graph = json.loads(line)
                    if len(graph[self.n_ident]) > 500:
                        continue
                    example = DataEntry(datset=self, num_nodes=len(graph[self.n_ident]),
                                        features=graph[self.n_ident],
                                        edges=graph[self.g_ident], target=graph['target'][0])
                    self.test_examples.append(example)
    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)

