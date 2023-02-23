import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

class Preprocess:
    def __init__(self) -> None:
        input = open(r"C:\Users\rohan_naik\Desktop\semicolons23_feed.back_backend\server\main\grammar\vocab.train",'r')
        file = input.read(10000000).split('\n')
        data = []
        labels = []
        for entry in file:
            item = entry.split('\t')
            if(len(item)==6):
                data.append(self.preprocess(item[4]))
                labels.append(self.preprocess(item[5]))
        self.df = pd.DataFrame()
        self.df['enc_input'] = data[0:20000]
        self.df['dec_input'] = labels[0:20000]
        

    def remove_spaces(self,text):
        text = re.sub(r" '(\w)",r"'\1",text)
        text = re.sub(r" \,",",",text)
        text = re.sub(r" \.+","",text)
        text = re.sub(r" \!+","!",text)
        text = re.sub(r" \?+","?",text)
        text = re.sub(" n't","n't",text)
        text = re.sub("[\(\)\;\_\^\`\/]","",text)
        return text

    def decontract(self,text):
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text

    def preprocess(self,text):
        text = self.remove_spaces(text)   # REMOVING UNWANTED SPACES
        text = re.sub(r"\.+",".",text)
        text = re.sub(r"\!+","!",text)
        text = self.decontract(text)    # DECONTRACTION
        text = re.sub("[^A-Za-z0-9 ]+","",text)
        text = text.lower()
        return text
    
    def get_tokenizers(self):
        tk_inp = Tokenizer()
        tk_inp.fit_on_texts(self.df.enc_input.apply(str))
        tk_out = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' )
        tk_out.fit_on_texts(self.df.dec_input.apply(str))
        return tk_inp,tk_out