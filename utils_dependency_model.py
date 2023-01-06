
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import numpy as np
from my_utils import text_to_num,task_to_num

from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
from torch import nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        for x in range(vocab.total_vocabs()):
            dep_tags[f'tag{x}'] = [vocab.word_to_indx[x][aux[i][x]] for i in range(len(aux))]


        #Junto todo en un nuevo dataset
        token_data[index] = {'text':text,
                                'input_ids':tokenized.input_ids,
                                'attention_mask':tokenized.attention_mask,
                                'labels':labels,
                                'dep_tags':dep_tags}

    return token_data




class TrainDataset(Dataset):
    def __init__(self,data,tasks):
        self.data = data
        self.tasks = tasks

    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        x = torch.tensor(self.data[index]['input_ids'])
        
        attention = torch.tensor(self.data[index]['attention_mask'])
        # Etiquetas de las dimensiones correspondientes
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

        
        dep_tags = []
        for item in self.data[index]['dep_tags']:
            dep_tags.append(self.data[index]['dep_tags'][item])


        sample = {'input_ids': x,
                'attention_mask': attention,
                'labels': labels,
                'dep_tags':torch.tensor(dep_tags),
                'num_vocabs':len(dep_tags) # Esto es porque necesito el dato para collate_fn
                }
        return  sample


    def __len__(self):
        return len(self.data)
# PENDIENTE: Crear una única clase dataset
class MyEvalDataSet(Dataset): # Version de evaluación, es casi identica a la version inicial de train que está
    #comentada
    def __init__(self,data,tasks,eval=False):
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



#######################################################################################
#
# COLLATE FUNCTION
#
#######################################################################################


# Tengo que crear mi propia función porque si no me da problemas con los tensores del parsing.
def collate_fn(batch):

    input_ids = [d['input_ids'] for d in batch]
    input_ids = pad_sequence(input_ids,batch_first=True)

    labels = [d['labels'] for d in batch]
    labels=torch.tensor(labels)


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
    

    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels.view(-1,len(labels)),'dep_tags': torch.tensor(deps)}


#######################################################################################
#
# CLASE VOCABULARY
#
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
        return len(self.word_to_indx)

    def _create_vocabulary(self,dataset,encoding):
        """Para otro tipo de encoding dará fallo, pendiente apliarlo"""
        if encoding == 'relative':
            num_splits = 2
            split_separator = '_'

        self.vocabs = {}
        for i in range(num_splits):
            self.vocabs[i] = set()


        for item in list(dataset.values()):
            for dep_label in item[encoding]:
                dep_label_split = dep_label.split(split_separator)

                
                for indx in range(num_splits):

                    self.vocabs[indx].add(dep_label_split[indx])

##############################################
# MODELO MULTITASK
##############################################

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






class MultiWithDependencies(nn.Module):
    """Modelo multitask que considera también los tags de dependencia"""
    def __init__(self,name,num_labels,tasks_names,vocab,embedding_dim=100,lstm_hidden_dim=128,dropout=0.1):
        super().__init__()
        self.num_labels = num_labels


        # Procesado texto
        self.encoder = AutoModel.from_pretrained(name,num_labels=num_labels,output_attentions=True,output_hidden_states = True)
        self.taskLayer = nn.ModuleList([])
        self.taskLayer.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768+lstm_hidden_dim,num_labels)
        ))


        
        self.task_to_num =  task_to_num

        
        if len(tasks_names) ==1:
            self.task = self.task_to_num[tasks_names[0]]
            self.SingleTask = True
        else:
            self.SingleTask=False
            self.tasksname = {v:k for v,k in enumerate(tasks_names)}
        
        # Procesado de los dependency tags
        
        self.LSTM_model = LSTM_enc(embedding_dim,lstm_hidden_dim,vocab)

    def forward(self,input_ids = None,attention_mask = None,labels = None,dep_tags = None,output_attentions=None,output_hidden_states=None):

        


        dBertoutputs = self.encoder(input_ids,attention_mask = attention_mask,output_attentions=output_attentions,output_hidden_states = output_hidden_states)





        outputs_last_hidden_state = dBertoutputs[0]

        cls_out = outputs_last_hidden_state[:,0]

        
        lstm_out , (lstm_hidden_state, cell_state) = self.LSTM_model(dep_tags)

        # Concateno ambos outputs
        output = torch.cat((cls_out,lstm_hidden_state.squeeze()),1)





        if self.SingleTask:
            tasks_output = output.clone()
            for layer in self.taskLayer:
                tasks_output = layer(tasks_output)

        else:
            tasks_output = {v : output.clone() for v in self.tasksname.keys()}

            for layer in self.taskLayer:
                tasks_output = {v: layer(k) for v,k in tasks_output.items()}





        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()


            if self.SingleTask:

                loss = loss_fct(tasks_output[0] , labels[0][:,self.task].type('torch.FloatTensor').to(device)) 
            else:

                task_loss = [loss_fct(tasks_output[i] , labels[i]) for i in range(len(tasks_output))]
                loss = sum(task_loss)

            


        return SequenceClassifierOutput(loss=loss, logits=tasks_output, hidden_states=dBertoutputs.hidden_states,attentions=dBertoutputs.attentions)