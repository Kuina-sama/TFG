
from tqdm.auto import tqdm 
from transformers import get_scheduler
import torch
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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



# Versión singletask

# def eval_function(model,metric,eval_dataloader):
#     '''Función para evaluar un modelo dado.
    
#     model: modelo a evaluar
#     metric: metrica a utilizar en la evaluación.
#     eval_dataloader: conjunto para evaluar. Debe ser un Pytorch DataLoader'''

#     model.eval()
#     for batch in eval_dataloader:
#         batch = {k:v.to(device) for k,v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
        
#         logits = outputs.logits 
#         predictions = torch.argmax(logits,dim=-1)
#         metric.add_batch(predictions = predictions, references = batch["labels"])

#     return metric.compute()


# Versión para un modelo planteado como multiTask. La evaluación sigue siendo individual, pero
# hay que indicar la tarea


def eval_function(model,metric,eval_dataloader,task = 0):
    '''Función para evaluar un modelo dado en la tarea indicada.
    
    model: modelo a evaluar
    metric: metrica a utilizar en la evaluación.
    eval_dataloader: conjunto para evaluar. Debe ser un Pytorch DataLoader. 
    task: tarea en la que queremos evaluar el modelo'''


    model.eval()
    for batch in eval_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        # logits = outputs.logits 
        logits = outputs.logits[task] 
        predictions = torch.argmax(logits,dim=-1)
        metric.add_batch(predictions = predictions, references = batch["labels"][:,0][:,task])

    return metric.compute()




# Clase MyDataSet. Crea dataset custom para poder usarlo luego para clasificación.

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self,data,labels,attention_mask):
        self.data = data
        self.labels =labels 
        self.attention = attention_mask


    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = self.data[index]
        labels =  self.labels[index]
        labels = np.array([labels])
        # labels = labels.astype('float')
        labels = labels
        attention = self.attention[index]


        sample = {'input_ids': x,
                'attention_mask': attention,
                'labels': torch.from_numpy(labels).view(-1,3)}

        return  sample


    def __len__(self):
        return len(self.data)
