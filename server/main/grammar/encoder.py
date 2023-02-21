import tensorflow as tf
from tensorflow.keras import layers

class Encoder(tf.keras.layers.Layer):

      '''
      Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
      '''
      
      def __init__(self, vocab_size,emb_dims, enc_units, input_length,batch_size):
          super().__init__()
          self.vocab_size = vocab_size
          self.emb_dims = emb_dims
          self.input_length = input_length
          # INITIALIZING THE REQUIRED VARIABLES
          self.batch_size=batch_size # BATHCH SIZE
          self.enc_units = enc_units # ENCODER UNITS

          # EMBEDDING LAYER
          self.embedding= layers.Embedding(vocab_size ,emb_dims) 
          # LSTM LAYER WITH RETURN SEQ AND RETURN STATES
          self.lstm = layers.LSTM(self.enc_units,return_state= True,return_sequences =  True) 
          
      def call(self, enc_input , states):
          '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
          '''
          # FORMING THE EMBEDDED VECTOR 
          emb = self.embedding(enc_input)
          # PASSING THE EMBEDDED VECTIO THROUGH LSTM LAYERS 
          enc_output,state_h,state_c = self.lstm(emb,initial_state=states)
          #RETURNING THE OUTPUT OF LSTM LAYER
          return enc_output,state_h,state_c 
      
      def initialize(self,batch_size):

          '''
          Given a batch size it will return intial hidden state and intial cell state.
          If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
          '''
          return tf.zeros(shape=(batch_size,self.enc_units)),tf.zeros(shape=(batch_size,self.enc_units))
      
      def get_config(self):
          config = super(Encoder, self).get_config()
          config.update({"vocab_size":self.vocab_size,"emb_dims":self.emb_dims, "enc_units":self.enc_units,"input_length":self.input_length,"batch_size":self.batch_size})
          return config