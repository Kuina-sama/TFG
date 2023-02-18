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
        return self.total_vocabs

    def _create_vocabulary(self,dataset,encoding):
        """Para otro tipo de encoding dará fallo, pendiente apliarlo"""
        if encoding == 'pos':
            self.total_vocabs = 3
            split_separator1 = '--'
            split_separator2 = '_'
        else: #Cojo dos vocabularios salvo en el caso del pos based
            self.total_vocabs = 2
            split_separator = '_'

        self.vocabs = {}
        for i in range(self.total_vocabs):
            self.vocabs[i] = set()

        if encoding == 'pos':
            for item in list(dataset.values()):
                for dep_label in item[encoding]:
                    split1, split2= dep_label.split(split_separator1)

                    split2, split3 = split2.split(split_separator2)
                    dep_label_split = (split1,split2,split3)
                    for indx in range(self.total_vocabs):

                        self.vocabs[indx].add(dep_label_split[indx])
                        self.vocabs[indx].add('unk')
           
        else:
            for item in list(dataset.values()):
                for dep_label in item[encoding]:
                    dep_label_split = dep_label.split(split_separator)

                    
                    for indx in range(self.total_vocabs):

                        self.vocabs[indx].add(dep_label_split[indx])
                        self.vocabs[indx].add('unk')
