#!/usr/bin/env python

"""
The implementation of the NNJM model with global context.
"""

from model_util import *
import io_read_ngram
import codecs
import theano
import theano.tensor as T
import numpy as np

class ModelGlobal(object):
    def __init__(self, local_context_size, in_vocab_size, emb_dim, hidden_sizes, act_func, n_out, pretrain_file, model_parameters, fix_emb, global_nonlinear):
        self.local_context_size = local_context_size
        self.in_vocab_size = in_vocab_size
        self.emb_dim = emb_dim
        self.hidden_sizes = hidden_sizes
        self.activation = get_activation_func(act_func)
        self.n_out = n_out
        self.pretrain_file = pretrain_file
        self.model_parameters = model_parameters
        self.global_nonlinear = global_nonlinear
        self.fix_emb = fix_emb
        
    def buildModel(self):
        ##############################
        # MODEL ARCHITECTURE

        # symbolic variables
        self.x = T.matrix('x')
        self.sm = T.matrix('sm')
        self.classifier = GlobalNNLM(self.x, self.sm, self.activation, self.in_vocab_size, self.emb_dim, self.hidden_sizes, self.n_out, self.local_context_size, 
            self.pretrain_file, self.model_parameters, self.fix_emb, self.global_nonlinear)
        self._initSGD()


    # CALCULATE GRAD USING SYMBOLIC FUNCTION
    def _initSGD(self):
        # symbolic variables
        y = T.ivector('y')
        lr = T.scalar('lr')

        # set symbolic cost function
        cost = self.classifier.nll(y)

        # set symbolic param gradients and adjust learning rate accordingly
        gparams = []
        grad_norm = 0.0
        for param in self.classifier.params:
            gparam = T.grad(cost, param)
            grad_norm += (gparam ** 2).sum()
            # gradients
            gparams.append(gparam)
        grad_norm = T.sqrt(grad_norm)
        max_grad_norm = 5
        if T.gt(grad_norm, max_grad_norm):
            lr = lr * max_grad_norm / grad_norm

        # set symbolic update rules
        updates = []
        for param, gparam in zip(self.classifier.params, gparams):
            updates.append((param, param - lr * gparam))
        
        self.y = y
        self.lr = lr
        self.cost = cost
        self.grad_norm = grad_norm
        self.updates = updates

    #################################################
    # used in train_util #####       BEGIN       ####
    #################################################
    def getTrainModel(self, data_x, data_y, data_sm):
        self.ngram_start_index = T.lscalar()
        self.ngram_end_index = T.lscalar()
        self.sm_start_index = T.lscalar()
        self.sm_end_index = T.lscalar()
        self.learning_rate = T.scalar()

        # TRAIN_MODEL
        self.train_outputs = [self.cost, self.grad_norm]
        self.train_set_x, self.train_set_y, self.train_set_sm = io_read_ngram.shared_dataset([data_x, data_y, data_sm])
        self.int_train_set_y = T.cast(self.train_set_y, 'int32')
        self.train_model = theano.function(inputs=[self.ngram_start_index, self.ngram_end_index, self.sm_start_index, self.sm_end_index, self.learning_rate], outputs=self.train_outputs, updates=self.updates,
                                      givens={
            self.x: self.train_set_x[self.ngram_start_index:self.ngram_end_index],
            self.y: self.int_train_set_y[self.ngram_start_index:self.ngram_end_index],
            self.sm: self.train_set_sm[self.sm_start_index:self.sm_end_index],
            self.lr: self.learning_rate})

        return self.train_model

    def getValidationModel(self, valid_set_x, valid_set_y, valid_set_sm, batch_size):
        self.num_valid_ngrams = valid_set_x.get_value(borrow=True).shape[0]
        self.num_valid_batches = (self.num_valid_ngrams - 1) / batch_size + 1
        self.valid_model = theano.function(inputs=[self.ngram_start_index, self.ngram_end_index, self.sm_start_index, self.sm_end_index], outputs=self.classifier.sum_ll(self.y),
                                      givens={
            self.x: valid_set_x[self.ngram_start_index:self.ngram_end_index],
            self.y: valid_set_y[self.ngram_start_index:self.ngram_end_index],
            self.sm: valid_set_sm[self.sm_start_index:self.sm_end_index],
            self.lr: self.learning_rate})

        return self.valid_model

    def getTestModel(self, test_set_x, test_set_y, test_set_sm, batch_size):
        self.num_test_ngrams = test_set_x.get_value(borrow=True).shape[0]
        self.num_test_batches = (self.num_test_ngrams - 1) / batch_size + 1
        self.test_model = theano.function(inputs=[self.ngram_start_index, self.ngram_end_index, self.sm_start_index, self.sm_end_index], outputs=self.classifier.sum_ll(self.y),
                                      givens={
            self.x: test_set_x[self.ngram_start_index:self.ngram_end_index],
            self.y: test_set_y[self.ngram_start_index:self.ngram_end_index],
            self.sm: test_set_sm[self.sm_start_index:self.sm_end_index],
            self.lr: self.learning_rate})

        return self.test_model


    def updateTrainModelInput(self, data_x, data_y, data_sm):
        if len(data_y) == 0:
            return False # EOF
        self.train_set_x.set_value(
            np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
        self.train_set_y.set_value(
            np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
        self.int_train_set_y = T.cast(self.train_set_y, 'int32')
        self.train_set_sm.set_value(
            np.asarray(data_sm, dtype=theano.config.floatX), borrow=True)
        return True

    # used in train_util: num_train_samples = model.getTrainSetXSize()
    def getTrainSetXSize(self):
        return self.train_set_x.get_value(borrow=True).shape[0]

    # for old interface
    def getModelSymbols(self):
        return (self.classifier, self.x, self.y, self.lr, self.cost, self.grad_norm, self.updates)
    #################################################
    # used in train_util #####       END         ####
    #################################################


class GlobalNNLM(object):

    def __init__(self, input_ngram, input_sm, activation, in_vocab_size, emb_dim, hidden_sizes, n_out, local_context_size, 
        pretrain_file, model_parameters, fix_emb, global_nonlinear):
        self.context_size = local_context_size + 1
        self.num_hidden_layers = len(hidden_sizes)
        self.emb_dim = emb_dim
        self.in_vocab_size = in_vocab_size

        # unpack the model parameters
        if model_parameters is not None:
            context_size, linear_W_emb, hidden_Ws, hidden_bs, softmax_W, softmax_b = model_parameters
            # model paramter validation check
            assert(context_size == local_context_size)
            assert(len(hidden_Ws) == self.num_hidden_layers)
        else:
            context_size, linear_W_emb, hidden_Ws, hidden_bs, softmax_W, softmax_b = None, None, None, None, None, None

        # linear embeding layer
        self.linearLayer = GlobalLinearLayer(input_ngram, input_sm, in_vocab_size, emb_dim, pretrain_file, 
            linear_W_emb=linear_W_emb, fix_emb=fix_emb, nonlinear=global_nonlinear, activation=activation)

        # hidden layers
        self.hidden_layers = []
        if global_nonlinear is None:
            hidden_in = emb_dim * self.context_size
        else:
            hidden_in = emb_dim * local_context_size + global_nonlinear
        hidden_params = []
        prev_layer = self.linearLayer
        for ii in xrange(self.num_hidden_layers):
            hidden_out = hidden_sizes[ii]
            hidden_layer = HiddenLayer(prev_layer.output, hidden_in, hidden_out, activation)
            self.hidden_layers.append(hidden_layer)
            hidden_params = hidden_params + hidden_layer.params
            hidden_in = hidden_out
            prev_layer = hidden_layer

        # softmax
        # for now do not pretrain softmax layer
        self.softmaxLayer = SoftmaxLayer(self.hidden_layers[len(hidden_sizes) - 1].output, hidden_out, n_out, None, None)

        # nll
        self.nll = self.softmaxLayer.nll

        # sum_ll
        self.sum_ll = self.softmaxLayer.sum_ll

        # params
        self.params = self.linearLayer.params + hidden_params + self.softmaxLayer.params
