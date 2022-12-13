
import stanza
import torch

import time
import sys 
sys.path.append(r"C:\Users\kuina\OneDrive\TFG\Codigo\CoDeLin")
from CoDeLin.utils.reader import parse_conllu
from CoDeLin.encs.enc_deps import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from tqdm import tqdm
from stanza.utils.conll import CoNLL


nlp = stanza.Pipeline(lang='en',processors='tokenize,mwt,pos,lemma,depparse',verbose=False, tokenize_no_ssplit=True)



def create_dependency_tags(dataset,encoding_type,separator):

    if encoding_type == 'absolute':
        encoder = naive_absolute.D_NaiveAbsoluteEncoding(separator)
    elif encoding_type == 'relative':
        encoder = naive_relative.D_NaiveRelativeEncoding(separator)
    elif encoding_type == 'pos':
        encoder = pos_based.D_PosBasedEncoding(separator)
    else:
        print(encoding_type, ' is not a valid type.')
        return 


    all_sentences = list(dataset.keys())

    print('Comenzando parsing de dependencias....\n')
    start = time.time()
    docs_in = [stanza.Document([], text = doc) for doc in all_sentences]

    docs_out = nlp(docs_in)
    end = time.time()
    print(f'Parsing de dependencias terminado. Duración del proceso: {(end-start)/60} minutos')
    print('Comenzando a generar el encoding para las dependencias')
    for item in tqdm(docs_out):

        CoNLL.write_doc2conll(item,'test.conllu') 
        with open('test.conllu','r') as f:
            node_list = parse_conllu(f)

        dataset[item.text][encoding_type] = [str(label) for label in encoder.encode(node_list[0])]


    print('Proceso terminado con éxito')

    return 
