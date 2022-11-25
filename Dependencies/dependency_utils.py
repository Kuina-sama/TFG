from supar import Parser
import stanza
import torch

import time
import sys 
sys.path.append(r"C:\Users\kuina\OneDrive\TFG\Codigo\CoDeLin")
from CoDeLin.utils.reader import parse_conllu
from CoDeLin.encs.enc_deps import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



parser = Parser.load('biaffine-dep-en')
nlp = stanza.Pipeline(lang='en',processors='tokenize,pos',verbose=False)



def create_dependency_conllu(dataset,output_file):
    """Toma un dataset que tiene por clave las
    oraciones a analizar y realiza las siguientes operaciones sobre cada oración:

    -Obtiene los tokens y pos-tags
    -Parsing de dependencias
    
    El resultado es guardado en un conllu file en la ruta output_file."""

    # En la prueba realizada tarda unos 27 min
    all_sentences = list(dataset.keys())
    print('Comenzando a generar fichero...')

    conllu_sentences = []
    start = time.time()
    for sentence in all_sentences:
        doc = nlp(sentence)
        conllu_sentences.append([(word.text,'_',word.upos) for sent in doc.sentences for word in sent.words ]) 
    
    parser.predict(conllu_sentences,pred=output_file,verbose=False)

    end = time.time()
    print('Proceso terminado: El archivo conllu ha sido creado en un tiempo de:  ', (end-start)/60)


def create_dependency_labels(data,conllu_file_path,encoding_type,separator):
    """Codifica el parsing de dependencias del conllu file según lo indicado en encoding_type.
    Devuelve todo en forma de diccionario"""

    if encoding_type == 'absolute':
        encoder = naive_absolute.D_NaiveAbsoluteEncoding(separator)
    elif encoding_type == 'relative':
        encoder = naive_relative.D_NaiveRelativeEncoding(separator)
    elif encoding_type == 'pos':
        encoder = pos_based.D_PosBasedEncoding(separator)
    else:
        print(encoding_type, ' is not a valid type.')
        return 

    with open(conllu_file_path,'r') as f:
        conllu_nodes = parse_conllu(f)


    data_dependency = {}
    for text, node_list in zip(data,conllu_nodes):
        # data[text]['dependency_tags'] =  [str(label) for label in encoder.encode(node_list)]
        data_dependency[text] = [str(label) for label in encoder.encode(node_list)]

    # return data
    return data_dependency
