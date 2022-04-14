""" BERT classifier for hate speech detection using HuggingFace transformers, Keras, and Tensorflow
@author Michael Miller Yoder, modified from Lynnette Hui Xian Ng
@year 2022
"""
import re
import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd

from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import keras
from keras.models import Model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def preprocess(text):
    """ Preprocess text """
    if text == '':
        return ''
    else:
        text = text.lower()
        text_cleaned = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text_cleaned = re.sub(r'#[A-Za-z0-9_]+', '', text_cleaned)
        text_cleaned = re.sub(r'https?:\/\/\S*', '', text_cleaned)
        text_cleaned = text_cleaned.replace(',', '')
    return text_cleaned


class BertClassifier:
    """ BERT hate speech classifier """

    def __init__(self):
        """ Initialize classifier """
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.callbacks = [
                #tf.keras.callbacks.ModelCheckpoint(
                #            filepath='../models/output',save_weights_only=True, monitor='val_loss', mode='min',save_best_only=True),
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
                        ]
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, epsilon=1e-8) #5e-7, 5e-9 previous settings
        self.epochs = 40

    def create_sentence_embeddings(self, sentences):
        input_ids=[]
        attention_masks=[]

        for sent in sentences:
            #bert_inp = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, 
            #        padding='max_length', return_attention_mask = True)
            # Not sure why this ^ replacement doesn't work
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bert_inp = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, 
                        pad_to_max_length=True, return_attention_mask = True, truncation='longest_first')
                input_ids.append(bert_inp['input_ids'])
                attention_masks.append(bert_inp['attention_mask'])
                                                        
        input_ids=np.asarray(input_ids)
        attention_masks=np.array(attention_masks)
        return input_ids, attention_masks

    def compile_fit(self, input_ids, attention_masks, labels):
        """ Compile and fit the model """
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        self.model.fit([input_ids, attention_masks], labels, batch_size=32, epochs=self.epochs, 
            callbacks=self.callbacks)
        #self.model.fit([input_ids, attention_masks], labels, batch_size=32, epochs=self.epochs, 
        #    callbacks=self.callbacks, validation_data=([input_ids_val, attention_masks_val], labels_val))

    def predict(self, input_ids, attention_masks, gold):
        """ Make predictions on test data 
            Returns predictions and a classification report dataframe
        """
        preds = self.model.predict([input_ids, attention_masks],batch_size=32)['logits'].argmax(axis=1)
        scores = pd.DataFrame(classification_report(gold, preds, output_dict=True))
        return scores, preds
