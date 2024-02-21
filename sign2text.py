import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import gelu
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping

def transformer_model(input_vocab_size, target_vocab_size, max_sequence_length, d_model=128, num_heads=4, num_encoder_layers=4, num_decoder_layers=4, dff=512, dropout_rate=0.1):
    batch_size = 32
    encoder_input = Input(shape=(max_sequence_length,), dtype=tf.int32, name='encoder_input')
    encoder_input_expanded = tf.repeat(encoder_input, repeats=batch_size, axis=0)
    encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)(encoder_input_expanded)
    encoder_output = encoder_embedding
    for i in range(num_encoder_layers):
        encoder_output = encoder_layer(d_model, num_heads, dff, dropout_rate)(encoder_output)

    decoder_input = Input(shape=(None,), dtype=tf.int32, name='decoder_input')
    decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)(decoder_input)
    decoder_output = decoder_embedding
    for i in range(num_decoder_layers):
        decoder_output = decoder_layer(d_model, num_heads, dff, dropout_rate)(decoder_output, encoder_output)

    output = Dense(target_vocab_size, activation='softmax')(decoder_output)

    model = transformer_model(input_vocab_size, target_vocab_size, max_sequence_length, encoder_input=encoder_input_expanded)
    return model

def encoder_layer(d_model, num_heads, dff, dropout_rate):
    inputs = Input(shape=(None, d_model))
    attn_output, _ = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = tf.keras.layers.Dense(dff, activation=gelu)(out1)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return Model(inputs=inputs, outputs=out2)

def decoder_layer(d_model, num_heads, dff, dropout_rate):
    dec_inputs = Input(shape=(None, d_model))
    enc_outputs = Input(shape=(None, d_model))
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask)(dec_inputs)
    attn1, attn_weights_block1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(dec_inputs, dec_inputs, attention_mask=look_ahead_mask)
    attn1 = Dropout(dropout_rate)(attn1)
    out1 = LayerNormalization(epsilon=1e-6)(attn1 + dec_inputs)
    attn2, attn_weights_block2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(out1, enc_outputs)
    attn2 = Dropout(dropout_rate)(attn2)
    out2 = LayerNormalization(epsilon=1e-6)(attn2 + out1)
    ffn_output = tf.keras.layers.Dense(dff, activation=gelu)(out2)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out3 = LayerNormalization(epsilon=1e-6)(ffn_output + out2)
    return Model(inputs=[dec_inputs, enc_outputs], outputs=out3)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

input_vocab_size = 15
target_vocab_size = 15
max_sequence_length = 30
model = transformer_model(input_vocab_size, target_vocab_size, max_sequence_length)

model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])

model.summary()
