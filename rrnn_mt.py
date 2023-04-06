from torch import nn 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np

#######################################################################################

# CREATE DATALOADER

#######################################################################################

class DatasetMultiTaskSimple(Dataset):
    def __init__(self,data,tasks,eval,deps=False):
        self.data = data
        self.tasks = tasks
        self.eval = eval
        self.deps = deps
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = torch.tensor(self.data[index]['input_ids'])



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

        if self.deps:
            dep_tags = []
            for item in self.data[index]['dep_tags']:
                dep_tags.append(self.data[index]['dep_tags'][item])

            sample = {'input_ids': x,
                    'dep_tags':torch.tensor(dep_tags),
                    'tasks':self.tasks,
                    'num_vocabs':len(dep_tags) # Esto es porque necesito el dato para collate_fn
                    }
        else:
            sample = {'input_ids': x,
                    'tasks':self.tasks
                    }

        sample.update(labels)

        return  sample
    

    def __len__(self):
        return len(self.data)
    

#######################################################################################

# collate_fn

#######################################################################################

def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)


    labels = {}
    for task in batch[0]['tasks']:
        labels[task] = torch.tensor([b[task][0] for b in batch])



    batched_input = {'input_ids':input_ids}
    batched_input.update(labels)

    return batched_input


def collate_fn_dep(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)

    labels = {}
    for task in batch[0]['tasks']:
        labels[task] = torch.tensor([b[task][0] for b in batch])



    num_vocabs = batch[0]['num_vocabs']
    batch_size = len(batch)

    dep_tags = [b['dep_tags'] for b in batch]

    aux_dep = {}
    for j in range(num_vocabs):
        aux_dep[j] = [dep_tags[i][j] for i in range(batch_size)]
        aux_dep[j] = pad_sequence(aux_dep[j],batch_first=True)

    deps = [aux_dep[i].tolist() for i in range(num_vocabs)]
    
    batched_input = {'input_ids':input_ids, 'dep_tags': torch.tensor(deps)}
    batched_input.update(labels)

    return batched_input

#######################################################################################

# MODELOS

#######################################################################################

class MultiTaskRRNN(nn.Module):
    def __init__(self,emb_dim,vocab_size,lstm_hidden_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tasks = ['to','as','about']

        self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim,self.lstm_hidden_dim,batch_first = True,bidirectional=True)

        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(self.lstm_hidden_dim*2,64),
            nn.Linear(64,1)
        ))


    def forward(self,input_ids = None):

        out_emb = self.emb(input_ids)

        lstm_out , (lstm_hidden_state, cell_state)  = self.lstm(out_emb)

        lstm_cat = torch.cat([lstm_hidden_state[0],lstm_hidden_state[1]],dim=1)



        tasks_output = {v : lstm_cat for v in self.tasks}

        for layer in self.taskLayer:
            tasks_output = {v: layer(k) for v,k in tasks_output.items()}

        return tasks_output
    

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
    

class MultiTaskRRNNDep(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""
    def __init__(self,emb_dim,dep_vocab,vocab_size,lstm_hidden_dim,dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.tasks = ['to','as','about']

        self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim,self.lstm_hidden_dim,batch_first = True,bidirectional=True)

            
        # Capas modelo
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.lstm_hidden_dim*4,64), #Porque la lstm de la rrnn tiene output *2, y la de dep *2 también
            nn.Linear(64,1)
            ))
        self.LSTM_model = LSTM_enc(self.emb_dim,self.lstm_hidden_dim,dep_vocab)

     

    def forward(self,input_ids = None,dep_tags = None):


        


        out_emb = self.emb(input_ids)

        lstm_out_text , (lstm_hidden_state_text, cell_state_text)  = self.lstm(out_emb)

        lstm_cat_text = torch.cat([lstm_hidden_state_text[0],lstm_hidden_state_text[1]],dim=1)
        
        lstm_out , (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)

        lstm_cat = torch.cat([lstm_hidden_state[0],lstm_hidden_state[1]],dim=1)
        # Concateno ambos outputs
        output = torch.cat((lstm_cat_text,lstm_cat),1)

        tasks_output = {v : output for v in self.tasks}


        task_output = output

        for layer in self.taskLayer:
            tasks_output = {v: layer(k) for v,k in tasks_output.items()}

        return tasks_output