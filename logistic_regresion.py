from torch import nn 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np

from transformers import get_scheduler

from tqdm.auto import tqdm 

import utils_generic as generic
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def collate_fn(batch):

    input_ids = [b['input_ids'] for b in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)
    max_len = len(input_ids[0])
    batch_size = len(batch)
    label = torch.tensor([b['label'] for b in batch])


    
    batched_input = {'input_ids':input_ids, 'label': label,'max_len':max_len,'batch_size':batch_size}

    return batched_input


class LogisticRegression(nn.Module):
    def __init__(self,sequence_max_len):
        super().__init__()
        self.sequence_max_len = sequence_max_len
        # self.emb = nn.Embedding(self.vocab_size,self.emb_dim)
        self.linear = nn.Linear(self.sequence_max_len,1)
    
    def forward(self,batch):
        # out = self.linear(input_ids.float())
        input_ids = batch['input_ids']
        max_len = batch['max_len']
        batch_size = batch['batch_size']
        zero = torch.zeros((batch_size,self.sequence_max_len-max_len))
        x = torch.cat([input_ids,zero],dim=1)
        out = self.linear(x)
        return out


def validation_func(model,dl_val):
    model.eval()

    loss_fct = nn.BCEWithLogitsLoss()
    val_loss = 0
    for batch in dl_val:

        # batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():

            val_output = model(batch)
            loss = loss_fct(val_output , batch['label'].float().unsqueeze(1))


        val_loss += loss.item()

  
    return val_loss/len(dl_val)

def train_one_epoch(model,dl_train,optimizer,lr_scheduler,progress_bar):

    loss_fct = nn.BCEWithLogitsLoss()
    epoch_loss = 0
    for batch in dl_train:
        # batch = {k:v.to(device) for k,v in batch.items()} # Manda los datos a la gpu

        optimizer.zero_grad()

        output = model(batch)

        loss = loss_fct(output , batch['label'].float().unsqueeze(1))

        epoch_loss += loss.item()

        loss.backward() # Lo usa para calcular los gradientes

        optimizer.step() # training step en el optimizador
        lr_scheduler.step() # update del learning rate

        progress_bar.update(1)

    return epoch_loss


def train_function(model,num_epochs,dl_train,optimizer,early_stop = 10,dl_val = None,save_path='model'):
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

    # model.to(device)

    model.train()

    train_loss = []
    val_loss = []
    epochs_with_no_improvement = 0
    best_loss = 1

    for epoch in range(num_epochs):
        epoch_loss = 0 
        model.train()

        epoch_loss = train_one_epoch(model,dl_train,optimizer,lr_scheduler,progress_bar)
                
            
        epoch_train_loss =epoch_loss/len(dl_train)
        train_loss.append(epoch_train_loss)

        if dl_val:
            epoch_val_loss = validation_func(model,dl_val)
            val_loss.append(epoch_val_loss)
            print(f'Epoch {epoch+1} \t Training loss: {epoch_train_loss} \t Validation loss: {epoch_val_loss} \t ')
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                epochs_with_no_improvement = 0
                torch.save(model.state_dict(),save_path)
            # elif epoch_val_loss - best_loss >= 0.001:
            elif epoch_val_loss - best_loss >= 0.000001:
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

