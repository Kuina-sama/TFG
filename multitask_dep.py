import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import numpy as np


from transformers import AutoModel


import torch
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


from tqdm.auto import tqdm 
from transformers import get_scheduler
import torch
from datasets import load_metric

import matplotlib.pyplot as plt


from utils_generic import text_to_num, task_to_num



#######################################################################################

# CREATE DATALOADER

#######################################################################################


class DatasetMultitaskDep(Dataset):
    def __init__(self,data,tasks,eval):
        self.data = data
        self.tasks = tasks
        self.eval = eval

    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = torch.tensor(self.data[index]['input_ids'])
        
        attention = torch.tensor(self.data[index]['attention_mask'])
        
        # Etiquetas de las dimensiones correspondientes
        raw_labels = self.data[index]['labels'] 


        labels={'to':[], 'as':[],'about':[]}
        for task in self.tasks:
            aux = raw_labels[task]
            if len(aux)>1 :
                label = np.random.choice(aux)
                if label ==2:
                    label = np.random.choice([0,1])
                labels[task].append(label)
            elif len(aux)==1:
                if aux[0] == 2:

                    label = np.random.choice([0,1])
                    labels[task].append(label)
                else:
                    labels[task].append(aux[0])
            elif len(aux) == 0 and self.eval==True:
                labels[task].append(2)

        dep_tags = []
        for item in self.data[index]['dep_tags']:
            dep_tags.append(self.data[index]['dep_tags'][item])


        sample = {'input_ids': x,
                'attention_mask': attention,
                'dep_tags':torch.tensor(dep_tags),
                'tasks':self.tasks,
                'num_vocabs':len(dep_tags) # Esto es porque necesito el dato para collate_fn
                }
        sample.update(labels)        
        return  sample


    def __len__(self):
        return len(self.data)
    


def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)

    labels = {}
    for task in batch[0]['tasks']:
        labels[task] = torch.tensor([b[task][0] for b in batch])



    attention = [b['attention_mask'] for b in batch]
    attention_mask = pad_sequence(attention,batch_first=True)

    num_vocabs = batch[0]['num_vocabs']
    batch_size = len(batch)

    dep_tags = [b['dep_tags'] for b in batch]

    aux_dep = {}
    for j in range(num_vocabs):
        aux_dep[j] = [dep_tags[i][j] for i in range(batch_size)]
        aux_dep[j] = pad_sequence(aux_dep[j],batch_first=True)

    deps = [aux_dep[i].tolist() for i in range(num_vocabs)]
    
    batched_input = {'input_ids':input_ids, 'attention_mask':attention_mask, 'dep_tags': torch.tensor(deps)}
    batched_input.update(labels)

    return batched_input

#######################################################################################

# VOCABULARY

#######################################################################################


class Vocabulary():
    def __init__(self,data,encoding):
        self.encoding = encoding
        self._create_vocabulary(data,encoding)

        self.word_to_indx = []
        
        for i in self.vocabs:
            self.word_to_indx.append({word: i for i,word in enumerate(self.vocabs[i])})


    def get_vocab_sizes(self):
        vocabs_len = []
        for i in self.vocabs:
            vocabs_len.append(len(self.vocabs[i]))

        return vocabs_len

    def total_vocabs(self):
        return self.total_vocabs

    def _create_vocabulary(self,dataset,encoding):
        """Para otro tipo de encoding dará fallo, pendiente apliarlo"""
        if encoding == 'pos':
            self.total_vocabs = 3
            split_separator1 = '--'
            split_separator2 = '_'
        else: #Cojo dos vocabularios salvo en el caso del pos based
            self.total_vocabs = 2
            split_separator = '_'

        self.vocabs = {}
        for i in range(self.total_vocabs):
            self.vocabs[i] = set()

        if encoding == 'pos':
            for item in list(dataset.values()):
                for dep_label in item[encoding]:
                    split1, split2= dep_label.split(split_separator1)

                    split2, split3 = split2.split(split_separator2)
                    dep_label_split = (split1,split2,split3)
                    for indx in range(self.total_vocabs):

                        self.vocabs[indx].add(dep_label_split[indx])
                        self.vocabs[indx].add('unk')
           
        else:
            for item in list(dataset.values()):
                for dep_label in item[encoding]:
                    dep_label_split = dep_label.split(split_separator)

                    
                    for indx in range(self.total_vocabs):

                        self.vocabs[indx].add(dep_label_split[indx])
                        self.vocabs[indx].add('unk')


#######################################################################################

# MODELS

#######################################################################################


class LSTM_enc(nn.Module):
    """Modelo encargado de crear los embeddings para los tags de dependencia y contextualizarlos
    mediante el uso de una red LSTM.
    Funciona independientemente del número de items en los que separemos los dependency tags"""
    def __init__(self,embedding_dim,hidden_dim,vocab):
        """vocab: objeto de la clase Vocabulary"""
        super(LSTM_enc,self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_sizes = vocab.get_vocab_sizes()
        self.num_vocabs = len(self.vocab_sizes)


        self.emb_layers = nn.ModuleList([])
        for i in range(self.num_vocabs):
            self.emb_layers.append(nn.Embedding(self.vocab_sizes[i],embedding_dim)) 

        self.lstm = nn.LSTM(embedding_dim * self.num_vocabs,hidden_dim,batch_first = True,bidirectional=True)

    def forward(self,dep_tags):


        embeds = []

        for i in range(len(dep_tags)):

            e = self.emb_layers[i](dep_tags[i])

            embeds.append(e)


        concat_embeds = torch.cat(embeds,2)


        return self.lstm(concat_embeds)



class MultiWithDependencies(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""
    def __init__(self,conf,vocab,dropout=0.1):
        super().__init__()
        self.num_labels = conf['num_labels']
        self.name = conf['model_name']
        self.embedding_dim = conf['embedding_dim']
        self.lstm_hidden_dim = conf['lstm_hidden_dim']
        self.encoder_dim = conf['encoder_dim']

        self.tasks = ['to','as','about']
            
        # Capas modelo
        self.encoder = AutoModel.from_pretrained(self.name,num_labels=self.num_labels,output_attentions=True,output_hidden_states = True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder_dim+self.lstm_hidden_dim*2,1)
        ))
        self.LSTM_model = LSTM_enc(self.embedding_dim,self.lstm_hidden_dim,vocab)

     

    def forward(self,input_ids = None,attention_mask = None,dep_tags = None):


        dBertoutputs = self.encoder(input_ids,attention_mask = attention_mask,output_attentions=True,output_hidden_states = True)



        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:,0]

        
        lstm_out , (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)
        
        lstm_cat = torch.cat([lstm_hidden_state[0],lstm_hidden_state[1]],dim=1)
        # Concateno ambos outputs
        output = torch.cat((cls_out,lstm_cat.squeeze()),1)




        tasks_output = {v : output for v in self.tasks}

        for layer in self.taskLayer:
            tasks_output = {v: layer(k) for v,k in tasks_output.items()}
        


        return tasks_output
