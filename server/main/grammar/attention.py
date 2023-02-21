import tensorflow as tf
from tensorflow.keras import layers

class Monotonic_Attention(tf.keras.layers.Layer):
    '''THIS FUNCTION RETURNS THE CONTEXT VECTOR AND ATTENTION WEIGHTS (ALPHA VALUES)'''
    def __init__(self,units):
        super().__init__()
        self.units = units
        # INITIALIZING THE DENSE LAYER W1
        self.W1 = layers.Dense(units)
        # INITIALIZING THE DENSE LAYER W2
        self.W2 = layers.Dense(units)
        # INITIALIZING THE DENSE LAYER V
        self.v = layers.Dense(1)
        
    def call(self,enc_output,dec_state,prev_att):
        # HERE WE ARE COMPUTING THE SCORE 
        dec_state =  tf.expand_dims(dec_state,axis=-1)
        score = tf.matmul(enc_output,dec_state)
        score = tf.squeeze(score, [2])
        # AFTER THE SOCRES ARE COMPUTED THE SIGMOID IS USED ON IT
        probabilities = tf.sigmoid(score)
        # ATTENTION WEIGHTS FOR PRESENT TIME STEP
        probabilities = probabilities*tf.cumsum(tf.squeeze(prev_att,-1), axis=1)
        attention = probabilities*tf.math.cumprod(1-probabilities, axis=1, exclusive=True)
        attention = tf.expand_dims(attention,axis=-1)
        
        # CONTEXT VECTOR
        context_vec  =  attention  * enc_output
        context_vec = tf.reduce_sum(context_vec,axis=1)
        
        # RETURN CONTEXT VECTOR AND ATTENTION
        return context_vec, attention

    def get_config(self):
        config = super(Monotonic_Attention, self).get_config()
        config.update({"units":self.units})
        return config

    