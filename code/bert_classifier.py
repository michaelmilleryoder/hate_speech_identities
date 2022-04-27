""" BERT classifier for hate speech detection using HuggingFace transformers, Keras, and Tensorflow
@author Michael Miller Yoder, modified from Lynnette Hui Xian Ng
@year 2022
"""
import re
import warnings
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

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

    def __init__(self, n_gpus=1, gpu_ids='all'):
        """ Initialize classifier """
        #self.n_gpus = self.strategy.num_replicas_in_sync
        self.n_gpus = int(n_gpus)
        if gpu_ids != 'all':
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gid) for gid in gpu_ids])
        tf.keras.backend.clear_session()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.callbacks = [
                #tf.keras.callbacks.ModelCheckpoint(
                #            filepath='../models/output',save_weights_only=True, monitor='val_loss', mode='min',save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
                        ]
        self.epochs = 50 # maximum number of epochs to train
        #self.strategy = tf.distribute.MirroredStrategy() # for multiple GPUs
        #self.strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        if self.n_gpus > 1:
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
            with self.strategy.scope():
                self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, epsilon=1e-8) #5e-7, 5e-9 previous settings
        else:
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, epsilon=1e-8) #5e-7, 5e-9 previous settings
        self.batch_size_per_replica = 32
        self.batch_size = self.batch_size_per_replica * self.n_gpus
        #tqdm.write(f"Strategy uses {self.strategy.num_replicas_in_sync} GPUs")
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

    def train_eval(self, orig_train, test):
        """ Train and evaluate on train and test dataframes """

        # Get a dev set as 10% of train
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.1)
        for train_inds, dev_inds in splitter.split(orig_train, groups=orig_train.index):
            train = orig_train.iloc[train_inds]
            dev = orig_train.iloc[dev_inds]

        # Vectorize, preprocess input
        input_ids_train, attention_masks_train = self.create_sentence_embeddings(
                train['text'].map(preprocess))
        input_ids_dev, attention_masks_dev = self.create_sentence_embeddings(
                dev['text'].map(preprocess))
        input_ids_test, attention_masks_test = self.create_sentence_embeddings(
                test['text'].map(preprocess))

        # Train, evaluate model 
        self.build_compile_fit(input_ids_train, attention_masks_train, train['hate'], 
                input_ids_dev, attention_masks_dev, dev['hate'])
        fold_scores, preds = self.predict(input_ids_test, attention_masks_test, test['hate'])
        return fold_scores, preds                 

    def build_model(self):
        """ Define a model """
        if self.n_gpus > 1:
            with self.strategy.scope():
                self.model = TFBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
                self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])
        else:
            self.model = TFBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])

    def build_compile_fit(self, input_ids, attention_masks, labels, 
            input_ids_dev, attention_masks_dev, labels_dev):
        """ Specify, compile and fit the model """
        self.build_model()

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
                callbacks=self.callbacks, 
                validation_data=([input_ids_dev, attention_masks_dev], labels_dev),
                verbose=0)
        tf.keras.backend.clear_session()
        K.clear_session()

    def predict(self, input_ids, attention_masks, gold):
        """ Make predictions on test data 
            Returns predictions and a classification report dataframe
        """
        preds = self.model.predict([input_ids, attention_masks],batch_size=32)['logits'].argmax(axis=1)
        scores = pd.DataFrame(classification_report(gold, preds, output_dict=True))
        return scores, preds
