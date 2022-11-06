from CoDeLin.utils.reader import parse_conllu
from supar import Parser

import json

import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


parser = Parser.load('biaffine-dep-en')


def create_conllu_file(dataset,output):

    """output must be the relative path to the output file, something like: r'./wizard_completo.conllu'"""



    my_data = {}
    for key, value in enumerate(dataset['train']['text']):
        my_data[key] = {'text': value, 'encoded_labels': None}


    conllu_format={}
    for key in my_data:
        sentence = my_data[key]['text']


        conllu_format[key] = parser.predict(sentence,lang='en',verbose=False)[0]



    with open(output,'w',encoding='utf-8') as f:
        for key in my_data:
            text = my_data[key]
            conllu_item = conllu_format[key]

            f.write(f'# text = {text} \n{conllu_item}\n')


    return my_data




def create_parsing_labels_dataset(conllu_file,my_data,encoder):


    with open(conllu_file,'r',encoding='utf-8') as data_file:
        conllu_nodes = parse_conllu(data_file)

    for key, node_list in zip(my_data,conllu_nodes):
        my_data[key]['encoded_labels'] = [str(label) for label in encoder.encode(node_list)]


    return my_data

