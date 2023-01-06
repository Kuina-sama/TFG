
from tqdm.auto import tqdm 
from transformers import get_scheduler
import torch
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



#######################################################################################

text_to_num = {'to':{'PARTNER:female':0,'PARTNER:male':1,"PARTNER:unknown":2},
                'as':{'SELF:female':0, 'SELF:male':1,'SELF:unknown':2},
                'about':{'ABOUT:female':0,'ABOUT:male':1,'ABOUT:unknown':2}}

num_to_text = {'to':{0:'PARTNER:female',1:'PARTNER:male',2:"PARTNER:unknown"},
                'as':{0:'SELF:female', 1:'SELF:male',2:'SELF:unknown'},
                'about':{0:'ABOUT:female',1:'ABOUT:male',2:'ABOUT:unknown'}}

all_tasks_names = ['about','as','to']


task_to_num =  {'about':0,'as':1,'to':2}


#######################################################################################

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


    for epoch in range(num_epochs): 
        for batch in train_dataloader: 
            batch = {k:v.to(device) for k,v in batch.items()} # Manda los datos a la gpu

            outputs = model(**batch) # Calcula outputs


            loss = outputs.loss 
            loss.backward() # Lo usa para calcular los gradientes

            optimizer.step() # training step en el optimizador
            lr_scheduler.step() # update del learning rate
            optimizer.zero_grad()
            progress_bar.update(1)
        print(progress_bar)


    return





# def eval_function_multi(model,eval_dataloader,tasks,metric):
#     """Función para evaluar modelo.
#     Considera una única métrica indicada en el parámetro metric.
#     tasks: Tareas en las que evaluaré el modelo. Si el modelo
#     no tiene datos de esas tareas lanzará un error"""
#     model.eval()
#     metrics = {task : metric for task  in range(len(tasks))} # Se podria modificar en funcion del tipo de tarea
#     for batch in eval_dataloader:
#         batch = {k:v.to(device) for k,v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)

#         logits = outputs.logits
#         predictions = {task : torch.argmax(logits[task],dim=-1) for task in range(len(tasks))}

#         for task, metric in metrics.items():
#             mask = batch["labels"][:,0][:,task] != 2
#             if len(predictions[1][mask]) == 0:
#                 continue
                
#             metric.add_batch(predictions = predictions[task][mask], references = batch["labels"][:,0][:,task][mask])


#     return {tasks[task] : metric.compute() for task, metric in metrics.items()}
from datasets import load_metric
def eval_function_multi(model,eval_dataloader,tasks_names):

    model.eval()
    metrics = {task : load_metric("accuracy") for task  in range(len(tasks_names))} # Se podria modificar en funcion del tipo de tarea
    for batch in eval_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        

        logits = outputs.logits
        predictions = {task : torch.argmax(logits[task],dim=-1) for task in range(len(tasks_names))}

        for task, metric in metrics.items():
            mask = batch["labels"][:,0][:,task] != 2
            if len(predictions[1][mask]) == 0:
                continue
                
            metric.add_batch(predictions = predictions[task][mask], references = batch["labels"][:,0][:,task][mask])


    return {tasks_names[task] : metric.compute() for task, metric in metrics.items()}
