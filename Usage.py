import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

preprocessor = hub.KerasLayer('FinBERT/preprocess_layer')
bert_model = hub.KerasLayer('FinBERT/bert_layer')

emb_vec = bert_model(preprocessor(text_input))['pooled_output']
output_layer = tf.keras.layers.Dense(len(classes), activation='sigmoid')(emb_vec)

model = tf.keras.models.Model(text_input, output_layer)
