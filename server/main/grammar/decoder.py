import tensorflow as tf
from tensorflow.keras import layers
from .attention import Monotonic_Attention

class Onestepdecoder(tf.keras.Model):
    '''THIS MODEL OUTPUTS THE RESULT OF DECODER FOR ONE TIME SETP GIVEN THE INPUT FOR PRECIOVE TIME STEP'''

    def __init__(self, vocab_size,emb_dims, dec_units, input_len,att_units,batch_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.dec_units = dec_units
        self.input_len = input_len
        self.att_units = att_units
        self.batch_size = batch_size
        # INTITALIZING THE REQUIRED VARIABLES
        # EMBEDDING LAYERS
        self.emb = layers.Embedding(vocab_size,emb_dims,input_length= input_len)
        # ATTENTION LAYER
        self.att = Monotonic_Attention(att_units)
        # LSTM LAYER
        self.lstm = layers.LSTM(dec_units,return_sequences=True,return_state=True)
        # DENSE LAYER
        self.dense = layers.Dense(vocab_size,activation="softmax")

    def call(self, encoder_output , input , state_h,state_c,previous_attention):
        # FORMING THE EMBEDDED VECTOR FOR THE WORD
        # (32,1)=>(32,1,12)
        emb = self.emb(input)
        dec_output,dec_state_h,dec_state_c = self.lstm(emb, initial_state = [state_h,state_c] )
        # GETTING THE CONTEXT VECTOR AND ATTENTION WEIGHTS BASED ON THE ENCODER OUTPUT AND  DECODER STATE_H
        context_vec,alphas = self.att(encoder_output,dec_state_h,previous_attention)
        # CONCATINATING THE CONTEXT VECTOR(BY EXPANDING DIMENSION) AND ENBEDDED VECTOR
        dense_input =  tf.concat([tf.expand_dims(context_vec,1),dec_output],axis=-1)
        # PASSING THE DECODER OUTPUT THROUGH DENSE LAYER WITH UNITS EQUAL TO VOCAB SIZE
        fc = self.dense(dense_input)
        # RETURNING THE OUTPUT
        return fc , dec_state_h , dec_state_c , alphas


    def get_config(self):
        config=(
            { 
                "vocab_size":self.vocab_size,
                "emb_dims":self.emb_dims,
                "dec_units": self.dec_units,
                "input_len": self.input_len,
                "att_units":self.att_units,
                "batch_size":self.batch_size
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Decoder(tf.keras.Model):
    '''THIS MODEL PERFORMS THE WHOLE DECODER OPERATION FOR THE COMPLETE SENTENCE'''
    def __init__(self, vocab_size,emb_dims, dec_units, input_len,att_units,batch_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.dec_units = dec_units
        self.att_units = att_units
        self.batch_size = batch_size
        # INITIALIZING THE VARIABLES
        # LENGTH OF INPUT SENTENCE
        self.input_len = input_len
        # ONE STEP DECODER
        self.onestepdecoder = Onestepdecoder(vocab_size,emb_dims, dec_units, input_len,att_units,batch_size)

    def call(self,dec_input,enc_output,state_h,state_c,initial_attention):
        # THIS VATIABLE STORES THE VALUE OF STATE_H FOR THE PREVIOUS STATE
        current_state_h = state_h 
        current_state_c = state_c
        previous_attention = initial_attention
        # THIS STORES THE DECODER OUTPUT FOR EACH TIME STEP
        pred = []
        # THIS STORED THE ALPHA VALUES
        alpha_values = []
        # FOR EACH WORD IN THE INPUT SENTENCE
        for i in range(self.input_len):
            
            # CURRENT WORD TO INPUT TO ONE STEP DECODER
            current_vec = dec_input[:,i]

            # EXPANDING THE DIMENSION FOR THE WORD
            current_vec = tf.expand_dims(current_vec,axis=-1)

            # PERFORMING THE ONE STEP DECODER OPERATION 
            dec_output,dec_state_h,dec_state_c,alphas = self.onestepdecoder(enc_output ,current_vec,current_state_h,current_state_c,previous_attention)

            #UPDATING THE CURRENT STATE_H
            current_state_h = dec_state_h
            current_state_c = dec_state_c
            previous_attention = alphas
            
            #APPENDING THE DECODER OUTPUT TO "pred" LIST
            pred.append(dec_output)

            # APPENDING THE ALPHA VALUES
            alpha_values.append(alphas)
            
        # CONCATINATING ALL THE VALUES IN THE LIST
        output = tf.concat(pred,axis=1)
        # CONCATINATING ALL THE ALPHA VALUES IN THE LIST
        alpha_values = tf.concat(alpha_values,axis = -1)
        # RETURNING THE OUTPUT
        return output , alpha_values

    def get_config(self):
      config = ({ "vocab_size":self.vocab_size,"emb_dims":self.emb_dims,"dec_units": self.dec_units, "input_len":self.input_len,"att_units":self.att_units,"batch_size":self.batch_size})
      return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)