#!/usr/bin/env python

"""
Train a NNJM (with global context) model.
"""

usage = 'To train NNJM (with global context) using Theano'

import cPickle
import gzip
import os
import sys
import time
import re
import codecs
import argparse
import datetime

import numpy as np
import theano
import theano.tensor as T

# our libs
import model_global
import io_vocab
import io_read_ngram
import io_model
from train_util import *

def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """

    parser = argparse.ArgumentParser(description=usage)  # add description
    # positional arguments
    parser.add_argument(
        'train_file', metavar='train_file', type=str, help='train file')
    parser.add_argument(
        'valid_file', metavar='valid_file', type=str, help='valid file')
    parser.add_argument(
        'test_file', metavar='test_file', type=str, help='test file')
    parser.add_argument(
        'ngram_size', metavar='ngram_size', type=int, help='ngram size')
    parser.add_argument('sentence_vector_length',
                        metavar='sentence_vector_length', type=int, help='sentence vector length')
    parser.add_argument(
        'vocab_size', metavar='vocab_size', type=int, help='vocab size')
    parser.add_argument(
        'vocab_file', metavar='vocab_file', type=str, help='vocab file')

    # optional arguments
    parser.add_argument('--model_file', dest='model_file', type=str,
                        default='', help='load model from a file (default=\'\')')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int,
                        default=128, help='embedding dimension (default=128)')
    parser.add_argument('--hidden_layers', dest='hidden_layers', type=str,
                        default='512', help='hidden layers, e.g. 512-512 (default=512)')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=0.1, help='learning rate (default=0.1)')
    parser.add_argument('--chunk', dest='chunk', type=int, default=2000,
                        help='each time consider batch_size*chunk ngrams (default=2000)')
    parser.add_argument('--valid_freq', dest='valid_freq',
                        type=int, default=1000, help='valid freq (default=1000)')
    parser.add_argument('--option', dest='opt', type=int, default=0,
                        help='option: 0 -- predict last word, 1 -- predict middle word (default=0)')
    parser.add_argument('--act_func', dest='act_func', type=str, default='relu',
                        help='non-linear function: \'tanh\' or \'relu\' (default=\'relu\')')
    parser.add_argument('--finetune', dest='finetune', type=int, default=1,
                        help='after training for this number of epoches, start halving learning rate(default: 1)')
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=5,
                        help='number of epochs, i.e. how many times to go throught the training data (default: 5)')
    
    # joint model
    parser.add_argument('--src_window', dest='src_window', type=int,
                        default=5, help='src window for joint model (default=5)')
    parser.add_argument('--src_lang', dest='src_lang',
                        type=str, default='', help='src lang (default=\'\')')
    parser.add_argument('--tgt_lang', dest='tgt_lang',
                        type=str, default='', help='tgt_lang (default=\'\')')

    # pretraining file
    parser.add_argument('--pretrain_file', dest='pretrain_file', type=str,
                        default=None, help='pretrain file for linear_W_emb layer (default=None)')
    parser.add_argument('--load_model', dest='load_model_file', type=str, default=None, help='Load model parameters from a pre-trained model')
    parser.add_argument('--fix_emb', dest='fix_emb', action='store_true', default=False, help='Use pretrain model and fix the embedding matrix during the training process')

    # global non-linearity
    parser.add_argument('--global_nonlinear', dest='global_nonlinear', type=int, default=None, help="Add a non-linear layer after the global mean sum")

    # remove stopwords
    parser.add_argument('--rm_stopwords', dest='stopword_cutoff', type=int, default=-1, help="Remove stopwords from the global sentence vector")

    args = parser.parse_args()
    return args


class TrainGlobalModel(TrainModel):
    def loadModelParams(self, ngram_size, src_window, model, max_src_sent_length):
        self.ngram_size = ngram_size
        self.src_window = src_window
        self.model = model
        self.max_src_sent_length = max_src_sent_length
        self.model_param_loaded = True

    def loadGlobalModelParams(self, stopword_cutoff):
        self.stopword_cutoff = stopword_cutoff

    def loadValidSet(self, valid_data_package):
        self.valid_set_x, self.valid_set_y, self.valid_set_sm = valid_data_package
        self.shared_valid_set_x, self.shared_valid_set_y, self.shared_valid_set_sm = io_read_ngram.shared_dataset(valid_data_package)
        self.shared_valid_set_y = T.cast(self.shared_valid_set_y, 'int32')
        self.valid_set_loaded = True

    def loadTestSet(self, test_data_package):
        self.test_set_x, self.test_set_y, self.test_set_sm = test_data_package
        self.shared_test_set_x, self.shared_test_set_y, self.shared_test_set_sm = io_read_ngram.shared_dataset(test_data_package)
        self.shared_test_set_y = T.cast(self.shared_test_set_y, 'int32')
        self.test_set_loaded = True

    def loadBatchData(self, isInitialLoad=False):
        src_lang = self.src_lang
        tgt_lang = self.tgt_lang
        tgt_vocab_size = self.tgt_vocab_size
        ngram_size = self.ngram_size

        chunk_size = self.chunk_size
        src_window = self.src_window
        opt = self.opt

        (self.data_x, self.data_y, self.data_sm) = io_read_ngram.get_joint_ngrams_with_src_global_matrix(self.src_f, self.tgt_f, self.align_f, \
            max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, num_read_lines=chunk_size, stopword_cutoff=self.stopword_cutoff)

        if isInitialLoad == False:
            assert(type(self.model) == model_global.ModelGlobal)
            return self.model.updateTrainModelInput(self.data_x, self.data_y, self.data_sm)

    def displayFirstNExamples(self, n):
        if self.src_window < 0:
            return
        src_vocab, src_vocab_size = io_vocab.load_vocab(self.src_vocab_file)
        tgt_vocab, tgt_vocab_size = io_vocab.load_vocab(self.tgt_vocab_file)
        src_inverse_vocab = io_vocab.inverse_vocab(src_vocab)
        tgt_inverse_vocab = io_vocab.inverse_vocab(tgt_vocab)
        assert(n <= self.chunk_size)
        for i in xrange(n):
            example_x = self.data_x[i]
            example_y = self.data_y[i]
            sent_idx = example_x[-1]
            src_sent_vector = self.data_sm[sent_idx]
            src_sent_length = src_sent_vector[0]
            src_sent_vector = src_sent_vector[1:src_sent_length+1]
            src_window_vector = example_x[:self.src_window*2 + 1]
            tgt_gram_vector = example_x[self.src_window*2 + 1:-1]
            src_sent_words = io_vocab.getWordsFromIndeces(src_sent_vector, src_inverse_vocab, self.tgt_vocab_size)
            src_window_words = io_vocab.getWordsFromIndeces(src_window_vector, src_inverse_vocab, self.tgt_vocab_size)
            tgt_gram_words = io_vocab.getWordsFromIndeces(tgt_gram_vector, tgt_inverse_vocab, 0)

            output = ""
            count = 0
            for w in src_window_words:
                count += 1
                if count == self.src_window + 1:
                    output += "[" + w + "] "
                else:
                    output += w + " "
            output += "|| "
            output += " ".join(tgt_gram_words) + " "
            output += "===> " + tgt_inverse_vocab[example_y]
            output += " |||| "
            output += " ".join(src_sent_words) + " "
            print output

    def trainOnBatch(self, train_model, i, batch_size, num_train_batches, num_train_samples, learning_rate):
        ngram_start_id = i * batch_size
        ngram_end_id = (i + 1) * batch_size if i < (num_train_batches - 1) else num_train_samples
        sm_start_id, sm_end_id = io_read_ngram.get_sentence_matrix_range(self.data_x, ngram_start_id, ngram_end_id)
        outputs = train_model(ngram_start_id, ngram_end_id, sm_start_id, sm_end_id, learning_rate)
        return outputs

    def buildModels(self):
        assert(hasattr(self, 'model'))
        print "Getting train model ..."
        train_model = self.model.getTrainModel(self.data_x, self.data_y, self.data_sm)
        print "Getting validation model ..."
        valid_model = self.model.getValidationModel(self.shared_valid_set_x, self.shared_valid_set_y, self.shared_valid_set_sm, self.batch_size)
        print "Getting test model ..."
        test_model = self.model.getTestModel(self.shared_test_set_x, self.shared_test_set_y, self.shared_test_set_sm, self.batch_size)
        print "Going to start training now ..."
        return (train_model, valid_model, test_model)

    def validate(self, model, num_ngrams, batch_size, num_batches):
        """
        Return average negative log-likelihood
        """
        loss = 0.0
        for i in xrange(num_batches):
            ngram_start_id = i * batch_size
            ngram_end_id = (i + 1) * batch_size if i < (num_batches - 1) else num_ngrams
            sm_start_id, sm_end_id = io_read_ngram.get_sentence_matrix_range(self.valid_set_x, ngram_start_id, ngram_end_id)
            loss -= model(ngram_start_id, ngram_end_id, sm_start_id, sm_end_id)  # model returns sum log likelihood
        loss /= num_ngrams
        perp = np.exp(loss)
        return (loss, perp)

    def test(self, model, num_ngrams, batch_size, num_batches):
        """
        Return average negative log-likelihood
        """
        loss = 0.0
        for i in xrange(num_batches):
            ngram_start_id = i * batch_size
            ngram_end_id = (i + 1) * batch_size if i < (num_batches - 1) else num_ngrams
            sm_start_id, sm_end_id = io_read_ngram.get_sentence_matrix_range(self.test_set_x, ngram_start_id, ngram_end_id)
            loss -= model(ngram_start_id, ngram_end_id, sm_start_id, sm_end_id)  # model returns sum log likelihood
        loss /= num_ngrams
        perp = np.exp(loss)
        return (loss, perp)

if __name__ == '__main__':

    ####################################
    # READ IN PARAMETERS
    args = process_command_line()
    print "Process ID: %d" % (os.getpid())
    print_cml_args(args)

    batch_size = 128
    emb_dim = args.emb_dim  # 128
    hidden_sizes = [int(x) for x in re.split('-', args.hidden_layers)]
    train_file = args.train_file
    learning_rate = args.learning_rate
    ngram_size = args.ngram_size
    valid_freq = args.valid_freq
    opt = args.opt
    chunk_size = args.chunk
    act_func = args.act_func
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    vocab_size = args.vocab_size
    finetune_epoch = args.finetune
    n_epochs = args.n_epochs
    src_window = args.src_window

    pretrain_file = args.pretrain_file
    global_nonlinear = args.global_nonlinear
    stopword_cutoff = args.stopword_cutoff
    fix_emb = args.fix_emb

    assert src_lang != '' and tgt_lang != ''

    # all the global context (sentence) will be extended to this length to
    # ensure a uniform length
    max_src_sent_length = args.sentence_vector_length  # often around 100

    ####################################
    # LOAD VACAB
    # <words> is a list of words as in string
    # <vocab_map> is a dict mapping from word string to integer number of 1,2,...|Vocab|
    # <vocab_size> is the size of vocab == len(words) == len(vocab_map).

    src_vocab_file = args.vocab_file + '.' + \
        str(args.vocab_size) + '.vocab.' + src_lang
    tgt_vocab_file = args.vocab_file + '.' + \
        str(args.vocab_size) + '.vocab.' + tgt_lang
    (src_vocab_map, src_vocab_size) = io_vocab.load_vocab(
        src_vocab_file)
    (tgt_vocab_map, tgt_vocab_size) = io_vocab.load_vocab(
        tgt_vocab_file)

    #######################################
    # LOAD VALID NGRAMS, LOAD TEST NGRAMS
    # <valid_set_x> is a list of list, each of the list in valid_set_x is a n-gram of word, each word is represented by an integer
    #       for e.g. [128, 11, 13, 33, 17, 22, 0, 0, 11, 3]
    # <valid_set_y> is a list of integers each represent a next-word following the list of word in valid_set_x

    src_valid_file = args.valid_file + '.' + \
        str(args.vocab_size) + '.id.' + src_lang
    tgt_valid_file = args.valid_file + '.' + \
        str(args.vocab_size) + '.id.' + tgt_lang
    # valid_set_sm is the sentence matrix
    (valid_set_x, valid_set_y, valid_set_sm) = io_read_ngram.get_all_joint_ngrams_with_src_global_matrix(src_valid_file, tgt_valid_file, args.valid_file + '.align',
                                                                  max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, stopword_cutoff=stopword_cutoff)
    src_test_file = args.test_file + '.' + \
        str(args.vocab_size) + '.id.' + src_lang
    tgt_test_file = args.test_file + '.' + \
        str(args.vocab_size) + '.id.' + tgt_lang
    (test_set_x, test_set_y, test_set_sm) = io_read_ngram.get_all_joint_ngrams_with_src_global_matrix(src_test_file, tgt_test_file, args.test_file + '.align',
                                                                  max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, stopword_cutoff=stopword_cutoff)


    if src_window >= 0:
        local_context_size = 2 * src_window + ngram_size  # 2 * 5 + 5 = 15
    else:
        local_context_size = ngram_size - 1
    # global_context_size = max_src_sent_length

    in_vocab_size = src_vocab_size + tgt_vocab_size
    out_vocab_size = tgt_vocab_size

    # Load model
    if args.load_model_file is not None:
        model_parameters = io_model.load_model(args.load_model_file)
    else:
        model_parameters = None

    #####################################
    # BUILD MODEL

    print "Start modeling part..."
    nnjm_global_model = model_global.ModelGlobal(local_context_size, in_vocab_size, emb_dim, hidden_sizes, act_func, out_vocab_size, pretrain_file, model_parameters, fix_emb, global_nonlinear)
    nnjm_global_model.buildModel()
    # model = nnlm_model.getModelSymbols()

    ####################################
    # START TRAINING

    print "Start training part... (1/2: loading)"
    train_model = TrainGlobalModel()
    train_model.loadVocab(src_lang, tgt_lang, tgt_vocab_size, src_vocab_file, tgt_vocab_file)
    train_model.loadValidSet((valid_set_x, valid_set_y, valid_set_sm))
    train_model.loadTestSet((test_set_x, test_set_y, test_set_sm))
    train_model.loadModelParams(ngram_size, src_window, nnjm_global_model, max_src_sent_length)
    train_model.loadTrainParams(train_file, batch_size, learning_rate, opt, valid_freq, finetune_epoch, chunk_size, vocab_size, n_epochs)
    train_model.loadGlobalModelParams(stopword_cutoff)
    print "Start training part... (2/2: training)"
    train_model.train()
