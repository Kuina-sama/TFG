
import stanza
import torch

import time
import sys 
sys.path.append(r"C:\Users\kuina\OneDrive\TFG\Codigo\CoDeLin")
from CoDeLin.models.conll_node import ConllNode
from CoDeLin.encs.enc_deps import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from tqdm import tqdm



nlp = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma,depparse',verbose=False, tokenize_no_ssplit=True)



# def create_dependency_tags(dataset,encoding_type,separator):

#     if encoding_type == 'absolute':
#         encoder = naive_absolute.D_NaiveAbsoluteEncoding(separator)
#     elif encoding_type == 'relative':
#         encoder = naive_relative.D_NaiveRelativeEncoding(separator)
#     elif encoding_type == 'pos':
#         encoder = pos_based.D_PosBasedEncoding(separator)
#     else:
#         print(encoding_type, ' is not a valid type.')
#         return 


#     all_sentences = list(dataset.keys())

#     print('Comenzando parsing de dependencias....\n')
#     start = time.time()
#     docs_in = [stanza.Document([], text = doc) for doc in all_sentences]

#     docs_out = nlp(docs_in)
#     end = time.time()
#     print(f'Parsing de dependencias terminado. Duración del proceso: {(end-start)/60} minutos')
#     print('Comenzando a generar el encoding para las dependencias')
#     for doc in tqdm(docs_out):

#         dicts = doc.to_dict()
#         conllu_nodes = []
#         for item in dicts[0]:
#             id = item.get('id','_')
#             form = item.get('text','_')
#             lemma =  item.get('lemma','_')
#             upos =  item.get('upos','_')
#             xpos = item.get('xpos','_')
#             feats = item.get('feats','_')
#             head = item.get('head','_')
#             deprel = item.get('deprel','_')
            
#             conllu_nodes.append(ConllNode(wid = id, form = form, lemma =  lemma, upos =  upos, xpos= xpos, 
#                 feats = feats, head= head, deprel =  deprel, deps = '_', misc = '_'))


        
#         dataset[doc.text][encoding_type] = [str(label) for label in encoder.encode(conllu_nodes)]


#     print('Proceso terminado con éxito')

#     return 


def create_dependency_tags(dataset,encoding_type:list,separator:str):
    """supported encoders: absolute
        relative,
        pos"""

    encoders = {'absolute':naive_absolute.D_NaiveAbsoluteEncoding(separator),
    'relative':naive_relative.D_NaiveRelativeEncoding(separator),
    'pos':pos_based.D_PosBasedEncoding(separator)
    }



    all_sentences = list(dataset.keys())

    print('Comenzando parsing de dependencias....\n')
    start = time.time()
    docs_in = [stanza.Document([], text = doc) for doc in all_sentences]

    docs_out = nlp(docs_in)
    end = time.time()
    print(f'Parsing de dependencias terminado. Duración del proceso: {(end-start)/60} minutos')
    print('Comenzando a generar el encoding para las dependencias')

    index_errors = []
    for doc in tqdm(docs_out):

        dicts = doc.to_dict()
        conllu_nodes = []
        for item in dicts[0]:
            id = item.get('id','_')
            form = item.get('text','_')
            lemma =  item.get('lemma','_')
            upos =  item.get('upos','_')
            xpos = item.get('xpos','_')
            feats = item.get('feats','_')
            head = item.get('head','_')
            deprel = item.get('deprel','_')
            
            conllu_nodes.append(ConllNode(wid = id, form = form, lemma =  lemma, upos =  upos, xpos= xpos, 
                feats = feats, head= head, deprel =  deprel, deps = '_', misc = '_'))


        for type in encoding_type:
            try:
                dataset[doc.text][type] = [str(label) for label in encoders[type].encode(conllu_nodes)]
            except IndexError as e:
                index_errors.append((conllu_nodes,type))

    print('Proceso terminado con éxito')

    return index_errors

def process_md_dataset(data,tasks):
    """Formatea md_gender para que considere todas las tareas indicadas en tasks.
    Si un texto no tiene etiqueta para una tarea, tendrá asociada una lista vacía""" #El valor 
    #de relleno lo meto al crear el PytorchDataset


    new_data = {}
    map_md_tasks = {0:'label_about',1:'label_about',2:'label_to',3:'label_to',4:'label_as',5:'label_as'}

    for item in data:
        text = item['text']
        
        if text in new_data:
            pass
        else:
            new_data[text] = {f'label_{x}':[] for x in tasks}

        for label in item['labels']:
            task = map_md_tasks[label]
            if task in new_data[text]:
                new_data[text][task].append(data.features['labels'][0].int2str(label))




    return new_data
