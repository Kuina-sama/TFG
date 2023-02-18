import matplotlib.pyplot as plt


from transformers import AutoTokenizer
#######################################################################################

# CONSTANTS

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

# Funciones para formatear el dataset y tokenizar los textos

#######################################################################################

def tokenize_dataset(dataset,tasks_names,model_name):
    """Tokeniza y formatea el dataset indicado para las tareas indicadas en tasks_names.
    Usa el tokenizer propio del modelo indicado.

    NO considera la información de parsing de dependencias"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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


#######################################################################################

# PLOT FUNCTIONS

#######################################################################################
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



