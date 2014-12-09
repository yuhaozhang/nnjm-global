#!/usr/bin/env python

"""
The definition (modelling) and initialization of shared layers in the neural language model architecture.
"""

debug = False

import sys
import re
import codecs

import cPickle
import random
import numpy as np
import scipy.sparse as sp
import theano
import theano.tensor as T
import theano.sparse as S
reload(sys)
sys.setdefaultencoding("utf-8")

import io_vocab


rng = np.random.RandomState(1234)
init_range = 0.05

def rectifier(x):
    return x * (x > 0)


def leaky_rect(x):
    return x * (x > 0) + 0.01 * x * (x < 0)

def sigmoid(x):
    return T.nnet.sigmoid

def get_activation_func(act_func):
    if act_func == 'tanh':
        # sys.stderr.write('# act_func=tanh\n')
        activation = T.tanh
    elif act_func == 'relu':
        # sys.stderr.write('# act_func=rectifier\n')
        activation = rectifier
    elif act_func == 'leakyrelu':
        # sys.stderr.write('# act_func=leaky rectifier\n')
        activation = leaky_rect
    elif act_funct == 'sigmoid':
        activation = sigmoid
    else:
        sys.stderr.write(
            '! Unknown activation function %s, not tanh or relu\n' % (act_func))
        sys.exit(1)
    return activation



####################################
# class SOFTMAXLAYER
####################################


class SoftmaxLayer(object):

    """
    class SOFTMAXLAYER

    The class takes the output from the last hidden layer as input and compute the log probability of
    each possible next word in the vocab with a softmax function. Numerical stability is ensured.

    Argument:
      - input: a matrix where each row is output from last hidden layer, and row number equals to batch size

    Adapt from this tutorial http://deeplearning.net/tutorial/logreg.html
    """

    def __init__(self, input, in_size, out_size, softmax_W=None, softmax_b=None):
        global rng
        global init_range
        if softmax_W is None or softmax_b is None:
            # randomly intialize everything
            softmax_W = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(in_size, out_size)), dtype=theano.config.floatX)
            softmax_b = np.zeros((out_size,), dtype=theano.config.floatX)
        else:
            given_in_size, given_out_size = softmax_W.shape
            assert(given_in_size == in_size and given_out_size == out_size)

        # shared variables
        self.W = theano.shared(value=softmax_W, name='W', borrow=True)
        self.b = theano.shared(value=softmax_b, name='b', borrow=True)

        # compute the "z = theta * x" in traditional denotation
        # input: batch_size * hidden_dim
        # self.W: hidden_dim * |V|
        # self.b: 1 * |V|
        x = T.dot(input, self.W) + self.b

        # take max for numerical stability
        x_max = T.max(x, axis=1, keepdims=True)

        # Take the log of the denominator in the softmax function
        #   1. minus max from the original x so that numerical stability can be ensured when we take exp
        # 2. we can add the max back directly since we take log after
        # the exp
        self.log_norm = T.log(
            T.sum(T.exp(x - x_max), axis=1, keepdims=True)) + x_max

        # The log probability equals to the log numerator (theta * x)
        # minus the log of the denominator (log norm)
        self.log_p_y_given_x = x - self.log_norm

        # params
        self.params = [self.W, self.b]

    def nll(self, y):
        """
        Mean negative log-lilelihood
        """
        return -T.mean(self.log_p_y_given_x[T.arange(y.shape[0]), y])

    def sum_ll(self, y):
        """
        Sum log-lilelihood
        """
        return T.sum(self.log_p_y_given_x[T.arange(y.shape[0]), y])

    def ind_ll(self, y):
        """
        Individual log-lilelihood
        """
        return self.log_p_y_given_x[T.arange(y.shape[0]), y]

####################################
# class HIDDENLAYER
####################################


class HiddenLayer(object):

    def __init__(self, input, in_size, out_size, activation):
        """
        class HIDDENLAYER

        The class actually takes the input and then COMPUTE the output using activation func, W_values and b_values
        and stores the result in self.output.

        Argument:
          - input: a matrix where each row is the last layer ouput, and the row number equals to batch size
        """

        global rng
        global init_range
        # if hidden_W is None or hidden_b is None:
        #     # random initialize everything
        #     hidden_W = np.asarray(rng.uniform(
        #         low=-init_range, high=init_range, size=(in_size, out_size)), dtype=theano.config.floatX)
        #     hidden_b = np.zeros((out_size,), dtype=theano.config.floatX)
        # else:
        #     given_in_size, given_out_size = hidden_W.shape
        #     assert(given_out_size == out_size)
        #     assert(given_in_size <= in_size)
        #     random_in_size = in_size - given_in_size
        #     hidden_W_random = np.asarray(rng.uniform(
        #         low=-init_range, high=init_range, size=(random_in_size, out_size)), dtype=theano.config.floatX)
        #     hidden_W = np.concatenate(hidden_W, hidden_W_random, axis=0)

        hidden_W = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(in_size, out_size)), dtype=theano.config.floatX)
        hidden_b = np.zeros((out_size,), dtype=theano.config.floatX)

        # W
        self.W = theano.shared(value=hidden_W, name='W')

        # b
        self.b = theano.shared(value=hidden_b, name='b')

        # output
        self.output = activation(T.dot(input, self.W) + self.b)

        # params
        self.params = [self.W, self.b]

####################################
# class LINEARLAYER
####################################
def load_pretrain_emb(pretrain_file):
    f = file(pretrain_file, 'rb')
    linear_W_emb = cPickle.load(f)
    linear_W_emb = np.float32(linear_W_emb)
    return linear_W_emb

class LinearLayer(object):

    """
    class LINEARLAYER

    The linear layer used to compute the word embedding matrix. This class actually take the input
    and then COMPUTE the output using linear_W_emb and stores the result in self.output.

    Argument:
      - input: a matrix where each row is a ngram word vector, and row number equals to batch size.
    """

    # LINEAR LAYER MATRIX of dim(vocab_size, emb_dim)
    def __init__(self, input, vocab_size, emb_dim, pretrain_file, linear_W_emb=None):

        global rng
        global init_range
        if pretrain_file:
            linear_W_emb = load_pretrain_emb(pretrain_file)
            print "* Using pretrained linear_W_emb ..."
            assert(len(linear_W_emb) == vocab_size)
        else:
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')

        # stack vectors
        input = T.cast(input, 'int32')

        # output is a matrix where each row correponds to a ngram embedding vector, and row number equals to batch size
        # output dimensions: batch_size * (ngram_size * emb_dim)
        self.output = self.W_emb[input.flatten()].reshape(
            (input.shape[0], input.shape[1] * emb_dim))  # self.W_emb.shape[1]

        # params is the word embedding matrix
        self.params = [self.W_emb]


####################################
# class GLOBALLINEARLAYER
####################################
def addNonlinearLayer(input, in_size, out_size, activation):
    """
    Return a non-linear layer on top of the existing structure.
    """
    global rng
    global init_range

    global_W = np.asarray(rng.uniform(
            low=-init_range, high=init_range, size=(in_size, out_size)), dtype=theano.config.floatX)
    global_b = np.zeros((out_size,), dtype=theano.config.floatX)

    # W
    W = theano.shared(value=global_W, name='global_W')

    # b
    b = theano.shared(value=global_b, name='global_b')

    # output
    output = activation(T.dot(input, W) + b)

    # params
    params = [W, b]
    return (params, output)

class GlobalPlusLinearLayer(object):

    def __init__(self, input_ngram, input_sm, vocab_size, emb_dim, num_section, linear_W_emb=None, fix_emb=False, nonlinear=None, activation=None):
        
        global rng
        global init_range
        if linear_W_emb is None:
            # random initialize
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)
        else:
            # use the given model parameter
            given_vocab_size, given_emb_dim = linear_W_emb.shape
            assert(given_vocab_size == vocab_size and given_emb_dim == emb_dim)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')

        # stack vectors
        input_ngram = T.cast(input_ngram, 'int32')
        input_sm = T.cast(input_sm, 'int32')

        # output is a matrix where each row correponds to a context_size embedding vector, and row number equals to batch size
        # output dimensions: batch_size * ((context_size + 1) * emb_dim)
        output_local = self.W_emb[input_ngram[:, :-1].flatten()].reshape(
            (input_ngram.shape[0], emb_dim * (input_ngram.shape[1] - 1)))  # self.W_emb.shape[1]
        
        sentence_lengths = input_sm[:,0]
        sentence_matrix = input_sm[:,1:]

        sentence_num = sentence_matrix.shape[0]
        global_length = sentence_matrix.shape[1]
        section_length = T.cast(T.ceil(global_length / float(num_section)), 'int32')

        # For the first section
        sentence_embeddings = T.mean(self.W_emb[sentence_matrix[:, :section_length].flatten()].reshape(
            (sentence_num, section_length, emb_dim)), axis=1)

        # For the rest sections
        for i in xrange(1, num_section):
            current_section = T.mean(self.W_emb[sentence_matrix[:, i*section_length:(i+1)*section_length].flatten()].reshape(
                (sentence_num, section_length, emb_dim)), axis=1)
            sentence_embeddings = T.concatenate([sentence_embeddings, current_section], axis=1)

        # get the sentence index for each ngram vector, and transform it to 0-based
        sentence_indeces = input_ngram[:,-1]
        base_index = sentence_indeces[0]
        sentence_indeces = sentence_indeces - base_index

        # the last column of output should be a weighted sum of the sentence
        # vectors
        output_global = sentence_embeddings[sentence_indeces.flatten()].reshape((sentence_indeces.shape[0], emb_dim * num_section))

        # handle non-linear layer
        if nonlinear is None or activation is None:
            self.output = T.concatenate([output_local, output_global], axis=1)
            # params is the word embedding matrix
            self.params = [self.W_emb] if not fix_emb else []
        else:
            self.non_linear_params, non_linear_output_global = addNonlinearLayer(output_global, emb_dim * num_section, nonlinear, activation)
            self.output = T.concatenate([output_local, non_linear_output_global], axis=1)
            self.params = [self.W_emb] + self.non_linear_params if not fix_emb else self.non_linear_params


class GlobalPlusLinearLayerWithAdaptiveSplit(object):

    def __init__(self, input_ngram, input_sm, vocab_size, emb_dim, num_section, linear_W_emb=None, fix_emb=False, nonlinear=None, activation=None):
        
        global rng
        global init_range
        if linear_W_emb is None:
            # random initialize
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)
        else:
            # use the given model parameter
            given_vocab_size, given_emb_dim = linear_W_emb.shape
            assert(given_vocab_size == vocab_size and given_emb_dim == emb_dim)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')

        # stack vectors
        input_ngram = T.cast(input_ngram, 'int32')
        input_sm = T.cast(input_sm, 'int32')

        # output is a matrix where each row correponds to a context_size embedding vector, and row number equals to batch size
        # output dimensions: batch_size * ((context_size + 1) * emb_dim)
        output_local = self.W_emb[input_ngram[:, :-1].flatten()].reshape(
            (input_ngram.shape[0], emb_dim * (input_ngram.shape[1] - 1)))  # self.W_emb.shape[1]
        
        sentence_lengths = input_sm[:,0]

        # compute sentence embeddings (take the weighted sum)
        def weighted_sentence(sentence, sent_len, W):
            sec_length = T.cast(T.ceil(sent_len / float(num_section)), 'int32')

            # for every section except the last one
            for sec_num in xrange(num_section-1):
                sec_start_id = sec_num * sec_length
                sec_end_id = (sec_num+1) * sec_length
                sec_vector = T.mean(W[sentence[sec_start_id:sec_end_id].flatten()], axis=0)
                if sec_num == 0:
                    global_vector = sec_vector
                else:
                    global_vector = T.concatenate([global_vector, sec_vector], axis=0) # here is axis=0 because sec_vector is a vector
            # for the last section
            sec_start_id = (num_section - 1) * sec_length
            sec_end_id = sent_len
            # if sec_start_id >= sent_len, it means this section should contain 0 words, so use EOS embedding instead.
            sec_vector = T.switch(T.ge(sec_start_id, sent_len), W[io_vocab.VocabConstants.EOS_INDEX], T.mean(W[sentence[sec_start_id:sec_end_id].flatten()], axis=0))
            # num_section > 1
            global_vector = T.concatenate([global_vector, sec_vector], axis=0)
            global_vector_for_short = W[sentence[:num_section].flatten()].reshape((1, emb_dim * num_section))
            return T.switch(T.gt(num_section, sent_len), global_vector_for_short, global_vector)

        sentence_embeddings, updates = theano.scan(fn=weighted_sentence,
                                  outputs_info=None,
                                  sequences=[input_sm[:, 1:], sentence_lengths],
                                  non_sequences=[self.W_emb])

        # get the sentence index for each ngram vector, and transform it to 0-based
        sentence_indeces = input_ngram[:,-1]
        base_index = sentence_indeces[0]
        sentence_indeces = sentence_indeces - base_index

        # the last column of output should be a weighted sum of the sentence
        # vectors
        output_global = sentence_embeddings[sentence_indeces.flatten()].reshape((sentence_indeces.shape[0], emb_dim * num_section))

        # handle non-linear layer
        if nonlinear is None or activation is None:
            self.output = T.concatenate([output_local, output_global], axis=1)
            # params is the word embedding matrix
            self.params = [self.W_emb] if not fix_emb else []
        else:
            self.non_linear_params, non_linear_output_global = addNonlinearLayer(output_global, emb_dim * num_section, nonlinear, activation)
            self.output = T.concatenate([output_local, non_linear_output_global], axis=1)
            self.params = [self.W_emb] + self.non_linear_params if not fix_emb else self.non_linear_params


class GlobalLinearLayer(object):

    def __init__(self, input_ngram, input_sm, vocab_size, emb_dim, pretrain_file, linear_W_emb=None, fix_emb=False, nonlinear=None, activation=None):
        
        global rng
        global init_range
        if pretrain_file:
            linear_W_emb = load_pretrain_emb(pretrain_file)
            print "---> Using pretrained linear_W_emb, with shape: " + str(linear_W_emb.shape)
            assert(len(linear_W_emb) == vocab_size)
        elif linear_W_emb is None:
            # random initialize
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)
        else:
            # use the given model parameter
            given_vocab_size, given_emb_dim = linear_W_emb.shape
            assert(given_vocab_size == vocab_size and given_emb_dim == emb_dim)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')

        # stack vectors
        input_ngram = T.cast(input_ngram, 'int32')
        input_sm = T.cast(input_sm, 'int32')

        # output is a matrix where each row correponds to a context_size embedding vector, and row number equals to batch size
        # output dimensions: batch_size * ((context_size + 1) * emb_dim)
        output_local = self.W_emb[input_ngram[:, :-1].flatten()].reshape(
            (input_ngram.shape[0], emb_dim * (input_ngram.shape[1] - 1)))  # self.W_emb.shape[1]
        
        sentence_lengths = input_sm[:,0]

        # compute sentence embeddings (take the weighted sum)
        def weighted_sentence(sentence, sent_len, W):
            return T.mean(W[sentence[0:sent_len].flatten()], axis=0)

        sentence_embeddings, updates = theano.scan(fn=weighted_sentence,
                                  outputs_info=None,
                                  sequences=[input_sm[:, 1:], sentence_lengths],
                                  non_sequences=[self.W_emb])

        # get the sentence index for each ngram vector, and transform it to 0-based
        sentence_indeces = input_ngram[:,-1]
        base_index = sentence_indeces[0]
        sentence_indeces = sentence_indeces - base_index

        # the last column of output should be a weighted sum of the sentence
        # vectors
        output_global = sentence_embeddings[sentence_indeces.flatten()].reshape((sentence_indeces.shape[0], emb_dim))

        # handle non-linear layer
        if nonlinear is None or activation is None:
            self.output = T.concatenate([output_local, output_global], axis=1)
            # params is the word embedding matrix
            self.params = [self.W_emb] if not fix_emb else []
        else:
            self.non_linear_params, non_linear_output_global = addNonlinearLayer(output_global, emb_dim, nonlinear, activation)
            self.output = T.concatenate([output_local, non_linear_output_global], axis=1)
            self.params = [self.W_emb] + self.non_linear_params if not fix_emb else self.non_linear_params

class GlobalLinearLayerMeanAll(object):

    def __init__(self, input_ngram, input_sm, vocab_size, emb_dim, local_context_size, global_context_size):
        
        global rng
        global init_range
        if pretrain_file:
            linear_W_emb = load_pretrain_emb(pretrain_file)
            print "* Using pretrained linear_W_emb ..."
            assert(len(linear_W_emb) == vocab_size)
        else:
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')

        # stack vectors
        input_ngram = T.cast(input_ngram, 'int32')
        input_sm = T.cast(input_sm, 'int32')

        # output is a matrix where each row correponds to a context_size embedding vector, and row number equals to batch size
        # output dimensions: batch_size * ((context_size + 1) * emb_dim)
        output_local = self.W_emb[input_ngram[:, :local_context_size].flatten()].reshape(
            (input_ngram.shape[0], local_context_size * emb_dim))  # self.W_emb.shape[1]
        # the last column of output should be a weighted sum of the sentence
        # vectors
        output_global = T.mean(self.W_emb[input_ngram[:, local_context_size:].flatten()].reshape(
            (input_ngram.shape[0], global_context_size, emb_dim)), axis=1)

        self.output = T.concatenate([output_local, output_global], axis=1)

        # params is the word embedding matrix
        self.params = [self.W_emb]

class GlobalLinearLayerAccurate(object):

    def __init__(self, input, input_sm, vocab_size, emb_dim, local_context_size, global_context_size):
        
        # initialize W_emb
        global rng
        global init_range
        if pretrain_file:
            linear_W_emb = load_pretrain_emb(pretrain_file)
            print "* Using pretrained linear_W_emb ..."
            assert(len(linear_W_emb) == vocab_size)
        else:
            linear_W_emb = np.asarray(rng.uniform(
                low=-init_range, high=init_range, size=(vocab_size, emb_dim)), dtype=theano.config.floatX)

        # shared variables
        self.W_emb = theano.shared(value=linear_W_emb, name='W_emb')
        # stack vectors
        input = T.cast(input, 'int32')

        # output is a matrix where each row correponds to a context_size embedding vector, and row number equals to batch size
        # output dimensions: batch_size * ((context_size + 1) * emb_dim)
        output_local = self.W_emb[input[:, :local_context_size].flatten()].reshape(
            (input.shape[0], local_context_size * emb_dim))  # self.W_emb.shape[1]

        # define symbolic functions for calculating the mean of sentences
        W = T.matrix('W')
        eos_vector = T.vector('eos_vector')
        eos_vector = T.fill(T.zeros_like(input[0,local_context_size:]), io_vocab.VocabConstants.EOS_INDEX)
        
        def weighted_sentence(sentence, W, eos_vector):
            sent_len = T.sum(T.neq(sentence, eos_vector))
            return T.mean(W[sentence[:sent_len]], axis=0)

        output_global, updates = theano.scan(fn=weighted_sentence,
                                  outputs_info=None,
                                  sequences=input[:, local_context_size:],
                                  non_sequences=[self.W_emb, eos_vector])

        # concatenate local output and global output to form the output matrix
        self.output = T.concatenate([output_local, output_global], axis=1)

        # params is the word embedding matrix
        self.params = [self.W_emb]
