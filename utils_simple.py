
# Recopila todas las funciones y clases usadas para trabajar sin el parsing
# de dependencias. Incluye el modelo multitask

from my_utils import text_to_num,task_to_num
import numpy as np

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch.utils.data import Dataset
from torch import nn




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def tokenize_dataset(dataset,tasks_names,tokenizer):
    """Tokeniza y formatea el dataset indicado. NO considera la información 
    de parsing de dependencias"""
    token_data = {}
    for index, text in enumerate(dataset):
        tokenized = tokenizer(text,truncation=True)

        labels ={}
        for task in tasks_names:
            aux_label = [text_to_num[task][x] for x in dataset[text][f'label_{task}']]


            labels[task] = aux_label

        token_data[index] = {'text':text,
                                'input_ids':tokenized.input_ids,
                                'attention_mask':tokenized.attention_mask,
                                'labels':labels}

    
    return token_data

class CustomDatasetSimple(Dataset):
    def __init__(self,data,tasks,eval):
        self.data = data
        self.tasks = tasks
        self.eval = eval

    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = torch.tensor(self.data[index]['input_ids'])

        raw_labels = self.data[index]['labels'] 

        labels=[]
        for task in self.tasks:
            aux = raw_labels[task]
            if len(aux)>1:
                label = np.random.choice(aux)
                if label ==2:
                    label = np.random.choice([0,1])
                labels.append(label)
            elif len(aux)==1:
                if aux[0] == 2:

                    label = np.random.choice([0,1])
                    labels.append(label)
                else:
                    labels.append(aux[0])
            #añadi solo estas dos lineas. Me interesa el eval porque si en train es falso
            #quiero que lance fallo
            elif len(aux) == 0 and self.eval==True:
                labels.append(2)



        labels = torch.tensor(labels)
        
        attention = torch.tensor(self.data[index]['attention_mask'])


        sample = {'input_ids': x,
                'attention_mask': attention,
                'labels': labels.view(-1,len(labels))}

        return  sample


    def __len__(self):
        return len(self.data)



class MultiTask_simple(nn.Module):
    """Modelo Multitask SIN procesado de los tags de dependencia"""
    def __init__(self,name,num_labels,tasks_names,dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(name,num_labels=num_labels,output_attentions=True,output_hidden_states = True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768,num_labels)
        ))

        
        self.task_to_num =  task_to_num

        
        if len(tasks_names) ==1:
            self.task = self.task_to_num[tasks_names[0]]
            self.SingleTask = True
        else:
            self.SingleTask=False
            self.tasksname = {v:k for v,k in enumerate(tasks_names)}
        
    def forward(self,input_ids = None,attention_mask = None,labels = None,output_attentions=None,output_hidden_states=None):
        


        dBertoutputs = self.encoder(input_ids,attention_mask = attention_mask,output_attentions=output_attentions,output_hidden_states = output_hidden_states)





        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:,0]

        if self.SingleTask:
            tasks_output = cls_out.clone()
            for layer in self.taskLayer:
                tasks_output = layer(tasks_output)

        else:
            tasks_output = {v : cls_out.clone() for v in self.tasksname.keys()}

            for layer in self.taskLayer:
                tasks_output = {v: layer(k) for v,k in tasks_output.items()}





        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()


            if self.SingleTask:

                loss = loss_fct(tasks_output[0] , labels[0][:,self.task].type('torch.FloatTensor').to(device)) 
            else:
                task_loss = [loss_fct(tasks_output[i] , labels[:,0][:,i]) for i in range(len(tasks_output))]
                loss = sum(task_loss)

            


        return SequenceClassifierOutput(loss=loss, logits=tasks_output, hidden_states=dBertoutputs.hidden_states,attentions=dBertoutputs.attentions)
