# from utils_generic import text_to_num,task_to_num

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn

import numpy as np

from transformers import AutoModel

from tqdm.auto import tqdm 
from transformers import get_scheduler

from datasets import load_metric


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import utils_generic as generic

#######################################################################################

# CREATE DATALOADER

#######################################################################################

class DatasetMultiTaskSimple(Dataset):
    def __init__(self,data,tasks,eval):
        self.data = data
        self.tasks = tasks
        self.eval = eval

    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = torch.tensor(self.data[index]['input_ids'])

        attention = torch.tensor(self.data[index]['attention_mask'])


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


        sample = {'input_ids': x,
                'attention_mask': attention,
                'tasks':self.tasks}

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


    batched_input = {'input_ids':input_ids, 'attention_mask':attention_mask}
    batched_input.update(labels)

    return batched_input


#######################################################################################

#  MODELO SINGLETASK

#######################################################################################


class MultiTaskSimple(nn.Module):
    
    """Modelo multitask que considera también los tags de dependencia"""
    def __init__(self,conf):
        super().__init__()
        self.num_labels = conf['num_labels']
        self.name = conf['model_name']
        self.encoder_dim = conf['encoder_dim']
        self.dropout = conf.get('dropout',0.1)
        self.tasks = ['to','as','about']

        # Capas modelo
        self.encoder = AutoModel.from_pretrained(self.name,num_labels=self.num_labels,output_attentions=True,output_hidden_states = True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(self.dropout),
            # nn.Linear(self.encoder_dim,self.num_labels)
            nn.Linear(self.encoder_dim,1)
        ))

    def forward(self,input_ids = None,attention_mask = None):

        


        dBertoutputs = self.encoder(input_ids,attention_mask = attention_mask,output_attentions=True,output_hidden_states = True)





        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:,0]

        tasks_output = {v : cls_out for v in self.tasks}

        for layer in self.taskLayer:
            tasks_output = {v: layer(k) for v,k in tasks_output.items()}
            


        return tasks_output

#######################################################################################

# TRAIN FUNCTIONS

#######################################################################################


def validation_func(model,dl_val,loss_fct):
    model.eval()

    val_loss = 0
    for batch in dl_val:

        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            val_outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])
        
        
        loss = 0
        for task in val_outputs:
            targets = batch[task].type(torch.LongTensor).to(device)
            predicted = val_outputs[task].to(device)
            loss += loss_fct(predicted , targets)


        val_loss += loss.item()

  
    return val_loss/len(dl_val)



def train_function(model,num_epochs,dl_train,optimizer,dl_val = None):
    '''
    model: modelo a entrenar.
    num_epoch: ciclos de entrenamiento
    dl_train: conjunto de entrenamiento, debe ser un Pytorch DataLoader
    optimizer: optimizador usado durante el entrenamiento.
    
    Esta función usará un learning rate scheduler para ir cambiando el 
    learning rate.
    
    Usará gpu siempre que esté disponible.
    
    Esta función irá usando paralelamente un conjunto de validación para calcular el error'''


    num_training_steps = num_epochs * len(dl_train)

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
        for batch in dl_train: 
            # try:
            batch = {k:v.to(device) for k,v in batch.items()} # Manda los datos a la gpu

            optimizer.zero_grad()
                
            outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) # Calcula outputs

            loss_fct = nn.CrossEntropyLoss()

            loss = 0
            for task in outputs:

                targets = batch[task].type(torch.LongTensor).to(device)
                predicted = outputs[task].to(device)
                loss += loss_fct(predicted , targets)

            epoch_loss += loss.item()

            loss.backward() # Lo usa para calcular los gradientes

            optimizer.step() # training step en el optimizador
            lr_scheduler.step() # update del learning rate

            
            
            progress_bar.update(1)
            # except RuntimeError as e:

            #     print(e)
                
            
        epoch_train_loss =epoch_loss/len(dl_train)
        train_loss.append(epoch_train_loss)

        if dl_val:
            epoch_val_loss = validation_func(model,dl_val,loss_fct)
            val_loss.append(epoch_val_loss)
            print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} \t Validation loss: {epoch_val_loss} ')
        else:
            print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} ')

        print(progress_bar)


    return (train_loss, val_loss)



def train_function_early_stop(model,num_epochs,dl_train,optimizer,early_stop = 10,dl_val = None,save_path='model'):
    '''
    model: modelo a entrenar.
    num_epoch: ciclos de entrenamiento
    dl_train: conjunto de entrenamiento, debe ser un Pytorch DataLoader
    optimizer: optimizador usado durante el entrenamiento.
    
    Esta función usará un learning rate scheduler para ir cambiando el 
    learning rate.
    
    Usará gpu siempre que esté disponible.
    
    Esta función irá usando paralelamente un conjunto de validación para calcular el error'''


    num_training_steps = num_epochs * len(dl_train)

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
    epochs_with_no_improvement = 0
    best_loss = 10
    for epoch in range(num_epochs):
        epoch_loss = 0 
        model.train()
        for batch in dl_train: 
            batch = {k:v.to(device) for k,v in batch.items()} # Manda los datos a la gpu

            optimizer.zero_grad()
                
            outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask']) # Calcula outputs

            loss_fct = nn.CrossEntropyLoss()

            loss = 0
            for task in outputs:

                targets = batch[task].type(torch.LongTensor).to(device)
                predicted = outputs[task].to(device)
                loss += loss_fct(predicted , targets)

            epoch_loss += loss.item()

            loss.backward() # Lo usa para calcular los gradientes

            optimizer.step() # training step en el optimizador
            lr_scheduler.step() # update del learning rate

            
            
            progress_bar.update(1)

                
            
        epoch_train_loss =epoch_loss/len(dl_train)
        train_loss.append(epoch_train_loss)

        if dl_val:
            epoch_val_loss = validation_func(model,dl_val,loss_fct)
            val_loss.append(epoch_val_loss)
            print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} \t Validation loss: {epoch_val_loss} ')
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                epochs_with_no_improvement = 0
                torch.save(model.state_dict(),save_path)
            elif epoch_val_loss - best_loss >= 0.001:
                epochs_with_no_improvement += 1
                print(f"\n{epochs_with_no_improvement} epoch without improvement")
                if epochs_with_no_improvement >= early_stop:
                    print(f"Validation_loss hasn't improve in {early_stop} epoch. Stopping training after {epoch+1} epochs...")

                    break
        else:
            print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} ')

        print(progress_bar)


            
        
	
    generic.plot_losses_val(train_loss,val_loss)
    return 






#######################################################################################

# EVAL FUNCTIONS

#######################################################################################
def eval_function_multi(model,eval_dataloader,tasks_names,metric_name):

    model.eval()
    metrics = {task  : load_metric(metric_name) for task  in tasks_names} # Se podria modificar en funcion del tipo de tarea
    for batch in eval_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])
        


        predictions = {task : torch.argmax(outputs[task],dim=-1) for task in tasks_names}
        labels = {task: batch[task] for task in ['to','as','about']}
        
        for task, metric in metrics.items():

            mask = labels[task] != 2


            if len(predictions[task][mask]) == 0: # Caso de que no tengamos esa tarea en la lista
                continue
                

            metric.add_batch(predictions = predictions[task][mask], references = labels[task][mask])


    return {task : metric.compute() for task, metric in metrics.items()}


def eval_function_gender(model,eval_dataloader,tasks_names,metric_name,gender_name:str):
    """0: female
        1:male 
        2:unknown"""

    if gender_name == 'female':
        gender = 0
    elif gender_name == 'male':
        gender = 1
    else:
        gender = 2
        
    model.eval()
    metrics = {task  : load_metric(metric_name) for task  in tasks_names} # Se podria modificar en funcion del tipo de tarea
    for batch in eval_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'],attention_mask = batch['attention_mask'])
        


        predictions = {task : torch.argmax(outputs[task],dim=-1) for task in tasks_names}
        labels = {task: batch[task] for task in ['to','as','about']}
        
        for task, metric in metrics.items():

            mask = labels[task] == gender


            if len(predictions[task][mask]) == 0: # Caso de que no tengamos predicciones con esa etiqueta
                continue
                

            metric.add_batch(predictions = predictions[task][mask], references = labels[task][mask])


    return {task : metric.compute() for task, metric in metrics.items()}