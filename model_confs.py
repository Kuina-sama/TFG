from transformers import DistilBertTokenizer, BertTokenizer

distilbert_conf = {'model_name':"distilbert-base-cased",
                'num_labels':2,
                'encoder_dim':768,
                'lstm_hidden_dim': 128,
                'embedding_dim':100,
                'tokenizer':DistilBertTokenizer.from_pretrained("distilbert-base-cased",do_lower_case=False,do_basic_tokenize=False)} 


bert_conf = {'model_name':"bert-base-cased",
                'num_labels':2,
                'encoder_dim':768,
                'lstm_hidden_dim': 128,
                'embedding_dim':100,
                'tokenizer':BertTokenizer.from_pretrained("bert-base-cased",do_lower_case=False,do_basic_tokenize=False)} 


roberta_conf = {'model_name':"roberta-base",
                'num_labels':2,
                'encoder_dim':768,
                'lstm_hidden_dim': 128,
                'embedding_dim':100,} 



