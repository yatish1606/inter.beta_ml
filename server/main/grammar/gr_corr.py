import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from preprocess import Preprocess
import numpy as np
from  tensorflow.keras.preprocessing.sequence import pad_sequences

class encoder_decoder(tf.keras.Model):
    '''THIS MODEL COMBINES ALL THE LAYERS AND FORM IN ENCODER DECODER MODEL WITH ATTENTION MECHANISM'''
    def __init__(self,enc_vocab_size,enc_emb_dim,enc_units,enc_input_length,
            dec_vocab_size,dec_emb_dim,dec_units,dec_input_length ,att_units, batch_size,att_mode="normal"):
        # INITAILIZING ALL VARIABLES
        super().__init__()
        self.enc_vocab_size= enc_vocab_size
        self.enc_emb_dim=enc_emb_dim
        self.enc_units= enc_units
        self.enc_input_length =enc_input_length
        self.dec_vocab_size=dec_vocab_size
        self.dec_emb_dim=dec_emb_dim
        self.dec_units=dec_units
        self.dec_input_length =dec_input_length
        self.att_units=att_units
        self.att_mode=att_mode

        # BATCH SIZE
        self.batch_size = batch_size
        # INITIALIZING ENCODER LAYER
        self.encoder = Encoder(enc_vocab_size, enc_emb_dim,enc_units, enc_input_length,batch_size)
        # INITALIZING DECODER LAYER
        self.decoder = Decoder(dec_vocab_size ,dec_emb_dim,dec_units,dec_input_length ,att_units, batch_size)
        self.input_len = enc_input_length
            
    def call(self,data):
        # THE INPUT OF DATALOADER IS IN A LIST FORM FOR EACH BATCH IT GIVER TWO INPUTS
        # INPUT1 IS FOR ENCODER
        # INPUT2 IS FOR DECODER
        inp1 , inp2 = data
        # PASSING THE INPUT1 TO ENCODER LAYER
        enc_output, enc_state_h, enc_state_c = self.encoder(inp1,self.encoder.initialize(self.batch_size))
        # PASSING INPUT2 TO THE DECODER LAYER
        initial_attention = np.zeros(shape = (self.batch_size,self.input_len,1),dtype="float32")
        initial_attention[:,1] = 1 
        dec_output , alphas = self.decoder(inp2 , enc_output,enc_state_h,enc_state_c ,initial_attention)
        # THE OUTPUT OF MODEL IS ONLY DECODER OUTPUT THE ALPHA VALUES ARE IGNORED HERE
        return dec_output

    def get_config(self):
        config = (
            {
            "enc_vocab_size":self.enc_vocab_size, 
            "enc_emb_dim":self.enc_emb_dim,"enc_units":self.enc_units,"enc_input_length":self.enc_input_length,\
            "dec_vocab_size":self.dec_vocab_size,
            "dec_emb_dim":self.dec_emb_dim,
            "dec_units":self.dec_units,
            "dec_input_length":self.dec_input_length ,\
            "att_units":self.att_units, "batch_size":self.batch_size
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GrammarCorrection:
    def __init__(self,path) -> None:
        self.tk_inp,self.tk_out = Preprocess().get_tokenizers()
        self.model = encoder_decoder(enc_vocab_size=len(self.tk_inp.word_index)+1,
                         enc_emb_dim = 300,
                         enc_units=256,enc_input_length=35,
                         dec_vocab_size =len(self.tk_out.word_index)+1,
                         dec_emb_dim =300,
                         dec_units=256,
                         dec_input_length = 35,
                         att_units=256,
                         batch_size=512)
        self.model.compile(optimizer="adam",loss='sparse_categorical_crossentropy')
        self.model.load_weights(path)
    
    def infer(self,sentence):
        # forming integer sequences
        seq = self.tk_inp.texts_to_sequences([sentence])
        # padding the sequences
        seq = pad_sequences(seq,maxlen = 20 , padding="post")
        # generating the output from encoder
        enc_output,state_h,state_c= self.model.layers[2](seq)
        # placeholder for predicted output
        pred = []
        input_state = [state_h,state_c]
        # initailizing the vector for inputing to decoder
        current_vec = tf.ones((1,1))
        
        for _ in range(20): # for each word in the input
            # passing each word through decoder layer
            dec_output,dec_state_h,dec_state_c = self.model.layers[3](current_vec , input_state)
            # passing decoder output through dense  layer
            dense = self.model.layers[4](dec_output)
            # taking argmax and getting the word index and updating the current vector
            current_vec = np.argmax(dense ,axis = -1)
            # updating the decoder states
            input_state = [dec_state_h,dec_state_c]
            # getting the actual word from the vocab
            pred.append(self.tk_out.index_word[current_vec[0][0]])
            
            # if the actual word is <end> break the loop
            if self.tk_out.index_word[current_vec[0][0]]=="<end>":
                break
            
        return " ".join(pred)
    
    def predict(self,text_blob):
        pass