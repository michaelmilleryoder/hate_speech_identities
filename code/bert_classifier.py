""" BERT classifier for hate speech detection using HuggingFace transformers, Keras, and Tensorflow
@author Michael Miller Yoder, modified from Lynnette Hui Xian Ng
@year 2022
"""
import re
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

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
        self.callbacks = [
                #tf.keras.callbacks.ModelCheckpoint(
                #            filepath='../models/output',save_weights_only=True, monitor='val_loss', mode='min',save_best_only=True),
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
                        ]
        self.epochs = 20
        #self.strategy = tf.distribute.MirroredStrategy() # for multiple GPUs
        #self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
        self.n_gpus = self.strategy.num_replicas_in_sync
        self.batch_size_per_replica = 32
        self.batch_size = self.batch_size_per_replica * self.n_gpus
        #tqdm.write(f"Strategy uses {self.strategy.num_replicas_in_sync} GPUs")
        with self.strategy.scope():
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, epsilon=1e-8) #5e-7, 5e-9 previous settings
        self.model = None

    def create_sentence_embeddings(self, sentences):
        input_ids=[]
        attention_masks=[]

        for sent in sentences:
            #bert_inp = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, 
            #        padding='max_length', return_attention_mask = True)
            # Not sure why this ^ replacement of the padding variable doesn't work

            # This formulation works, but I can't make TF tensors out of it
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bert_inp = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, 
                        pad_to_max_length=True, return_attention_mask = True, truncation='longest_first')
                input_ids.append(bert_inp['input_ids'])
                attention_masks.append(bert_inp['attention_mask'])

            # Trying to get input IDs and attention masks that work as TF datasets
            #bert_inp = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=64, 
            #        padding='max_length', return_attention_mask = True, return_tensors='tf')
            #input_ids.append(bert_inp['input_ids'])
            #attention_masks.append(bert_inp['attention_mask'])
                                                        
        input_ids=np.asarray(input_ids)
        attention_masks=np.array(attention_masks)
        return input_ids, attention_masks

    def train_eval(self, train, test):
        """ Train and evaluate on train and test dataframes """

        # Vectorize, preprocess input
        input_ids_train, attention_masks_train = self.create_sentence_embeddings(
                train['text'].map(preprocess))
        input_ids_test, attention_masks_test = self.create_sentence_embeddings(
                test['text'].map(preprocess))

        # Train, evaluate model 
        self.build_compile_fit(input_ids_train, attention_masks_train, train['hate'])
        fold_scores, preds = self.predict(input_ids_test, attention_masks_test, test['hate'])
        return fold_scores, preds                 

    def build_compile_fit(self, input_ids, attention_masks, labels):
        """ Specify, compile and fit the model """
        # TODO: make build a separate function
        with self.strategy.scope():
            self.model = TFBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])

        # Wrap data to tf Dataset
        #train = tf.data.Dataset.from_tensor_slices(([input_ids, attention_masks], labels))

        ## Set batch size on Dataset objects
        #batch_size = 32
        #train = train.batch(batch_size)

        ## Disable AutoShard
        #options = tf.data.Options()
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        #train = train.with_options(options)

        #self.model.fit(train, epochs=self.epochs, callbacks=self.callbacks)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit([input_ids, attention_masks], labels, batch_size=self.batch_size, 
                epochs=self.epochs, 
                callbacks=self.callbacks, verbose=0)
        #self.model.fit([input_ids, attention_masks], labels, batch_size=32, epochs=self.epochs, 
        #    callbacks=self.callbacks, validation_data=([input_ids_val, attention_masks_val], labels_val))

    def predict(self, input_ids, attention_masks, gold):
        """ Make predictions on test data 
            Returns predictions and a classification report dataframe
        """
        preds = self.model.predict([input_ids, attention_masks],batch_size=32)['logits'].argmax(axis=1)
        scores = pd.DataFrame(classification_report(gold, preds, output_dict=True))
        return scores, preds
