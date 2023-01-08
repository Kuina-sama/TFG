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

#######################################################################################

text_to_num = {'to':{'PARTNER:female':0,'PARTNER:male':1,"PARTNER:unknown":2},
                'as':{'SELF:female':0, 'SELF:male':1,'SELF:unknown':2},
                'about':{'ABOUT:female':0,'ABOUT:male':1,'ABOUT:unknown':2}}

num_to_text = {'to':{0:'PARTNER:female',1:'PARTNER:male',2:"PARTNER:unknown"},
                'as':{0:'SELF:female', 1:'SELF:male',2:'SELF:unknown'},
                'about':{0:'ABOUT:female',1:'ABOUT:male',2:'ABOUT:unknown'}}

all_tasks_names = ['about','as','to']


task_to_num =  {'about':0,'as':1,'to':2}

num_to_task = {0:'about',1:'as',2:'to'}


#######################################################################################


#######################################################################################
#
# FUNCIONES PARA EL PROCESADO DE LOS DATOS
#
#######################################################################################


def tokenize_dataset_with_dependencies(dataset,tokenizer,tasks_names,vocab):
    """ Tokeniza y formatea mi dataset. SI considera la información de parsing de dependencias.
    Por el momento funciona para encoding relative"""
    token_data = {}
    for index, text in enumerate(dataset):
        tokenized = tokenizer(text,truncation=True)

        labels ={}
        for task in tasks_names:
            aux_label = [text_to_num[task][x] for x in dataset[text][f'label_{task}']]


            labels[task] = aux_label

        # Esta parte procesa los tags de dependencia
        aux = [x.split('_') for x in dataset[text][vocab.encoding]]

        dep_tags = {}
        for x in range(vocab.total_vocabs):
            dep_tags[f'tag{x}'] = [vocab.word_to_indx[x].get(aux[i][x],vocab.word_to_indx[x]['unk']) for i in range(len(aux))]


        #Junto todo en un nuevo dataset
        token_data[index] = {'text':text,
                                'input_ids':tokenized.input_ids,
                                'attention_mask':tokenized.attention_mask,
                                'labels':labels,
                                'dep_tags':dep_tags}

    return token_data



class CustomDataset(Dataset):
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


# Tengo que crear mi propia función porque si no me da problemas con los tensores del parsing.
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

    dep_tags = [d['dep_tags'] for d in batch]
    aux_dep = {}
    for j in range(num_vocabs):
        aux_dep[j] = [dep_tags[i][j] for i in range(batch_size)]
        aux_dep[j] = pad_sequence(aux_dep[j],batch_first=True)

    deps = [aux_dep[i].tolist() for i in range(num_vocabs)]
    
    batched_input = {'input_ids':input_ids, 'attention_mask':attention_mask, 'dep_tags': torch.tensor(deps)}
    batched_input.update(labels)
    return batched_input



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
        if encoding == 'relative':
            self.total_vocabs = 2
            split_separator = '_'

        self.vocabs = {}
        for i in range(self.total_vocabs):
            self.vocabs[i] = set()


        for item in list(dataset.values()):
            for dep_label in item[encoding]:
                dep_label_split = dep_label.split(split_separator)

                
                for indx in range(self.total_vocabs):

                    self.vocabs[indx].add(dep_label_split[indx])
                    self.vocabs[indx].add('unk')


def train_function(model,num_epochs,train_dataloader,optimizer):
    '''
    model: modelo a entrenar.
    num_epoch: ciclos de entrenamiento
    train_dataloader: conjunto de entrenamiento, debe ser un Pytorch DataLoader
    optimizer: optimizador usado durante el entrenamiento.
    
    Esta función usará un learning rate scheduler para ir cambiando el 
    learning rate.
    
    Usará gpu siempre que esté disponible.'''


    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.to(device)

    model.train()
    train_loss = []

    for epoch in range(num_epochs):
        epoch_loss = []  
        for batch in train_dataloader: 
            batch = {k:v.to(device) for k,v in batch.items()} # Manda los datos a la gpu

            optimizer.zero_grad()
            
            outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],dep_tags = batch['dep_tags']) # Calcula outputs

            loss_fct = nn.CrossEntropyLoss()

            task_loss = [loss_fct(outputs[task] , batch[task]) for task in outputs]
            loss = sum(task_loss)
            
            loss.backward() # Lo usa para calcular los gradientes

            optimizer.step() # training step en el optimizador
            lr_scheduler.step() # update del learning rate
            
            
            progress_bar.update(1)
        epoch_loss.append(loss.item())
        e_loss = sum(epoch_loss)/len(epoch_loss)
        train_loss.append(e_loss)
        print(f'Epoch {epoch+1} \t Training loss: {e_loss} ')
        print(progress_bar)


    return train_loss

def validation_func(model,dl_val,loss_fct):
    model.eval()

    val_loss = 0
    for batch in dl_val:

        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            val_outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],dep_tags = batch['dep_tags'])
        
        
        loss = 0
        for task in val_outputs:
            targets = batch[task].type(torch.LongTensor).to(device)
            predicted = val_outputs[task].to(device)
            loss += loss_fct(predicted , targets)
        # task_loss = [loss_fct(val_outputs[task] , batch[task]) for task in val_outputs]
        # batch_loss = sum(task_loss)

        val_loss += loss.item()

  
    return val_loss/len(dl_val)

def train_functionV(model,num_epochs,train_dataloader,dl_val,optimizer):
    '''
    model: modelo a entrenar.
    num_epoch: ciclos de entrenamiento
    train_dataloader: conjunto de entrenamiento, debe ser un Pytorch DataLoader
    optimizer: optimizador usado durante el entrenamiento.
    
    Esta función usará un learning rate scheduler para ir cambiando el 
    learning rate.
    
    Usará gpu siempre que esté disponible.'''


    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps = 0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.to(device)

    model.train()
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0 
        model.train()
        for batch in train_dataloader: 
            try:
                batch = {k:v.to(device) for k,v in batch.items()} # Manda los datos a la gpu

                optimizer.zero_grad()
                
                outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],dep_tags = batch['dep_tags']) # Calcula outputs

                loss_fct = nn.CrossEntropyLoss()

                loss = 0
                for task in outputs:
                    targets = batch[task].type(torch.LongTensor).to(device)
                    predicted = outputs[task].to(device)
                    loss += loss_fct(predicted , targets)
                # task_loss = [loss_fct(outputs[task] , batch[task]) for task in outputs]
                # loss = sum(task_loss)
                epoch_loss += loss.item()

                loss.backward() # Lo usa para calcular los gradientes

                optimizer.step() # training step en el optimizador
                lr_scheduler.step() # update del learning rate

                
                
                progress_bar.update(1)
            except RuntimeError as e:
                print(e)
                print
            
        epoch_train_loss =epoch_loss/len(train_dataloader)
        train_loss.append(epoch_train_loss)

        epoch_val_loss = validation_func(model,dl_val,loss_fct)
        val_loss.append(epoch_val_loss)
        
        print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} \t Validation loss: {epoch_val_loss} ')
        print(progress_bar)


    return train_loss, val_loss

def eval_function_multi(model,eval_dataloader,tasks_names,metric_name):

    model.eval()
    metrics = {task  : load_metric(metric_name) for task  in tasks_names} # Se podria modificar en funcion del tipo de tarea
    for batch in eval_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'],dep_tags = batch['dep_tags'])
        


        predictions = {task : torch.argmax(outputs[task],dim=-1) for task in tasks_names}
        labels = {task: batch[task] for task in ['to','as','about']}
        
        for task, metric in metrics.items():

            mask = labels[task] != 2


            if len(predictions[task][mask]) == 0: # Caso de que no tengamos esa tarea en la lista
                continue
                
            # metric.add_batch(predictions = predictions[task][mask], references = batch["labels"][:,0][:,task][mask])

            metric.add_batch(predictions = predictions[task][mask], references = labels[task][mask])


    return {task : metric.compute() for task, metric in metrics.items()}


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

        self.lstm = nn.LSTM(embedding_dim * self.num_vocabs,hidden_dim,batch_first = True)

    def forward(self,dep_tags):


        embeds = []

        for i in range(len(dep_tags)):

            e = self.emb_layers[i](dep_tags[i])

            embeds.append(e)


        concat_embeds = torch.cat(embeds,2)


        return self.lstm(concat_embeds)


task_to_num =  {'about':0,'as':1,'to':2}
class MultiWithDependencies(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""
    def __init__(self,name,num_labels,tasks_names,vocab,embedding_dim=100,lstm_hidden_dim=128,dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.task_to_num =  task_to_num

        if len(tasks_names) ==1:
            self.task = self.task_to_num[tasks_names[0]]
            self.SingleTask = True
        else:
            self.SingleTask=False
            self.tasksname = {self.task_to_num[t]:t for t in tasks_names}
            
        # Capas modelo
        self.encoder = AutoModel.from_pretrained(name,num_labels=num_labels,output_attentions=True,output_hidden_states = True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.dim+lstm_hidden_dim,num_labels)
        ))
        self.LSTM_model = LSTM_enc(embedding_dim,lstm_hidden_dim,vocab)

     

    def forward(self,input_ids = None,attention_mask = None,dep_tags = None):

        


        dBertoutputs = self.encoder(input_ids,attention_mask = attention_mask,output_attentions=True,output_hidden_states = True)





        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:,0]

        
        lstm_out , (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)

        # Concateno ambos outputs
        output = torch.cat((cls_out,lstm_hidden_state.squeeze()),1)





        if self.SingleTask:
            task_output = output
            for layer in self.taskLayer:
                tasks_output = layer(task_output)

        else:
            tasks_output = {v : output for v in self.tasksname.keys()}

            for layer in self.taskLayer:
                tasks_output = {self.tasksname[v]: layer(k) for v,k in tasks_output.items()}
            


        return tasks_output



def plot_losses_val(train_loss,val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train loss','validation loss'])
    plt.title('Train-Validation loss')
    plt.show()

    return

def plot_losses_train(train_loss):
    plt.plot(train_loss)
    plt.legend(['train loss'])
    plt.title('Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return