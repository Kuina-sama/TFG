from torch import nn 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np

class DatasetSingleTaskSimple(Dataset):
    def __init__(self,data,task,eval,deps=False):
        self.data = data
        self.task = task
        self.eval = eval
        self.deps = deps
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = torch.tensor(self.data[index]['input_ids'])



        raw_label = self.data[index]['labels'][self.task]


        if len(raw_label)>1 :
            label = np.random.choice(raw_label)
            if label ==2:
                label = np.random.choice([0,1])
        elif len(raw_label)==1:
            if raw_label[0] == 2:
                label = np.random.choice([0,1])
            else:
                label = raw_label[0]
        elif len(raw_label) == 0 and self.eval==True:
            label = 2  


        if self.deps:
            dep_tags = []
            for item in self.data[index]['dep_tags']:
                dep_tags.append(self.data[index]['dep_tags'][item])

            sample = {'input_ids': x,
                    'dep_tags':torch.tensor(dep_tags),
                    'label':label,
                    'tasks':self.task,
                    'num_vocabs':len(dep_tags) # Esto es porque necesito el dato para collate_fn
                    }
        else:
            sample = {'input_ids': x,
                    'tasks':self.task,
                    'label':label}


        return  sample
    

    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)


    label = torch.tensor([b['label'] for b in batch])



    batched_input = {'input_ids':input_ids, 'label': label}

    return batched_input



def collate_fn_dep(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)

    label = torch.tensor([b['label'] for b in batch])



    num_vocabs = batch[0]['num_vocabs']
    batch_size = len(batch)

    dep_tags = [b['dep_tags'] for b in batch]

    aux_dep = {}
    for j in range(num_vocabs):
        aux_dep[j] = [dep_tags[i][j] for i in range(batch_size)]
        aux_dep[j] = pad_sequence(aux_dep[j],batch_first=True)

    deps = [aux_dep[i].tolist() for i in range(num_vocabs)]
    
    batched_input = {'input_ids':input_ids, 'dep_tags': torch.tensor(deps),'label':label}

    return batched_input


#######################################################################################

# MODELOS

#######################################################################################

class SingleTaskRRNN(nn.Module):
    def __init__(self,emb_dim,vocab_size,lstm_hidden_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.lstm_hidden_dim = lstm_hidden_dim

        self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim,self.lstm_hidden_dim,batch_first = True,bidirectional=True)
        self.linear = nn.Linear(self.lstm_hidden_dim*2,64)
        self.linear2 = nn.Linear(64,1)

    def forward(self,input_ids = None):

        out_emb = self.emb(input_ids)

        lstm_out , (lstm_hidden_state, cell_state)  = self.lstm(out_emb)

        lstm_cat = torch.cat([lstm_hidden_state[0],lstm_hidden_state[1]],dim=1)


        linear_out1 = self.linear(lstm_cat)
        linear_out2 = self.linear2(linear_out1)

        return linear_out2
    

### Esta versión me sirve para w2v, decide que usar en función de los parámetros
# de entrada
class SingleTaskRRNN2(nn.Module):
    def __init__(self,emb_dim,vocab_size,lstm_hidden_dim,emb_weights = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.lstm_hidden_dim = lstm_hidden_dim

        if emb_weights is not None:
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb_weights))
            self.lstm = nn.LSTM(300,self.lstm_hidden_dim,batch_first = True,bidirectional=True)
            self.linear = nn.Linear(self.lstm_hidden_dim*2,64)
            self.linear2 = nn.Linear(64,1)
        else:
            self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
            self.lstm = nn.LSTM(self.emb_dim,self.lstm_hidden_dim,batch_first = True,bidirectional=True)
            self.linear = nn.Linear(self.lstm_hidden_dim*2,64)
            self.linear2 = nn.Linear(64,1)

    def forward(self,input_ids = None):

        out_emb = self.emb(input_ids)

        lstm_out , (lstm_hidden_state, cell_state)  = self.lstm(out_emb)

        lstm_cat = torch.cat([lstm_hidden_state[0],lstm_hidden_state[1]],dim=1)


        linear_out1 = self.linear(lstm_cat)
        linear_out2 = self.linear2(linear_out1)

        return linear_out2
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
    

class SingleTaskRRNNDep(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""
    def __init__(self,emb_dim,dep_vocab,vocab_size,lstm_hidden_dim,emb_weights=None,dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.lstm_hidden_dim = lstm_hidden_dim

        if emb_weights is not None:
            self.emb = nn.Embedding.from_pretrained(torch.from_numpy(emb_weights))
            self.lstm = nn.LSTM(300,self.lstm_hidden_dim,batch_first = True,bidirectional=True)
        else:
            self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
            self.lstm = nn.LSTM(self.emb_dim,self.lstm_hidden_dim,batch_first = True,bidirectional=True)

            
        # Capas modelo
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
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




        task_output = output

        for layer in self.taskLayer:
            task_output = layer(task_output)
        


        return task_output
    