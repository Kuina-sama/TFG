
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



        dataset[doc.text][encoding_type] = [str(label) for label in encoder.encode(conllu_nodes)]


    print('Proceso terminado con éxito')

    return 
