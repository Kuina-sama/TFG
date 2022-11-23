from CoDeLin.utils.reader import parse_conllu
from CoDeLin.encs.enc_deps import *

from supar import Parser

import stanza

import json

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = Parser.load('biaffine-dep-en')


def create_conllu_file(my_data,output):

    """
    Recibe un dataset de Huggingface como entrada y realiza el parsing de dependencia de las oraciones.
    
    El resultado obtenido será guardado en formato conllu en la ruta indicada en el parametro output.
    dataset: huggingface dataset. 
    output must be the relative path to the output file, example: r'./wizard_completo.conllu'"""



    # my_data = {}
    # for key, value in enumerate(data['train']['text']):
    #     my_data[key] = {'text': value, 'encoded_labels': None}

    conllu_format={}
    for key in my_data:
        sentence = my_data[key]['text']


        conllu_format[key] = parser.predict(sentence,lang='en',verbose=False)[0]



    with open(output,'w',encoding='utf-8') as f:
        for key in my_data:
            text = my_data[key]
            conllu_item = conllu_format[key]

            f.write(f'# text = {text} \n{conllu_item}\n')

    print('Your ConLLU file has been created in: ', output)
    # return
    return my_data



nlp = stanza.Pipeline(lang='en',processors='tokenize,pos')
def get_pos_tag(sentence):

    # nlp = stanza.Pipeline(lang='en',processors='tokenize,pos')

    doc = nlp(sentence)

    pos_tags = [word.upos for sent in doc.sentences for word in sent.words ]

    return pos_tags

def create_conllu_file_with_pos(my_data,output):


    ###################
    # TO DO: Unir las dos funciones de create_conllu_filw
    ###################
    """Creates ConLLu file including pos_tag"""


    # my_data = {}
    # for key, value in enumerate(data['train']['text']):
    #     my_data[key] = {'text': value, 'encoded_labels': None}

    conllu_format={}
    for key in my_data:
        sentence = my_data[key]['text']


        parsed_sentence= parser.predict(sentence,lang='en',verbose=False)[0]



        parsed_sentence.values[3] = get_pos_tag(sentence)
        conllu_format[key] = parsed_sentence
        # print(f'Sentence number {key} added to conllu')

    with open(output,'w',encoding='utf-8') as f:
        for key in my_data:
            text = my_data[key]['text']

            conllu_item = conllu_format[key]

            f.write(f'# text = {text} \n{conllu_item}\n')

    print('Your ConLLU file has been created in: ', output)
    return my_data


def create_parsing_labels_dataset(conllu_file,my_data,encoding_type,separator):

    """Takes a conllu file and encode it's dependencies.
    Separator= separator used in your conllu file. Usuarlly _
    Encoding_type:
        -absolute: D_NaiveAbsoluteEncoding
        -relative: D_NaiveRelativeEncoding
        
        in any other case it will use absolute encoding by default"""

    if encoding_type == 'absolute':
        encoder = naive_absolute.D_NaiveAbsoluteEncoding(separator)
    elif encoding_type == 'relative':
        encoder = naive_relative.D_NaiveRelativeEncoding(separator)
    else:
        print(encoding_type, ' is not a valid type.')
        return 

    with open(conllu_file,'r',encoding='utf-8') as data_file:
        conllu_nodes = parse_conllu(data_file)

    for key, node_list in zip(my_data,conllu_nodes):
        my_data[key]['encoded_labels'] = [str(label) for label in encoder.encode(node_list)]


    return my_data
