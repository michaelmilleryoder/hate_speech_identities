""" BERT classifier for hate speech detection using HuggingFace transformers, Keras, and Tensorflow
@author Michael Miller Yoder, modified from Lynnette Hui Xian Ng
@year 2022
"""

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification

import keras
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


class BertClassifier:
    """ BERT hate speech classifier """

    def __init__():
        """ Initialize classifier """
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.callbacks = [tf.keras.callbacks.ModelCheckpoint(
                            filepath='../models/output',save_weights_only=True, monitor='val_loss', mode='min',save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
                        ]
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-7, epsilon=5e-9)

    def create_sentence_embeddings(self, sentences):
        input_ids=[]
        attention_masks=[]

        for sent in sentences:
            bert_inp = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, pad_to_max_length=True,
                                                                                    return_attention_mask = True)
            input_ids.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])
                                                        
        input_ids=np.asarray(input_ids)
        attention_masks=np.array(attention_masks)
        return input_ids, attention_masks

    def compile_fit(self, input_ids, attention_masks, labels, input_ids_val, attention_masks_val, labels_val):
        """ Compile and fit the model """
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        self.model.fit([input_ids, attention_masks], labels, batch_size=32, epochs=100, 
            callbacks=self.callbacks, validation_data=([input_ids_val, attention_masks_val], labels_val))
