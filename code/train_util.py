#!/usr/bin/env python

"""
The implementation of the shared model training code.
"""

import sys
import datetime
import time
import codecs
import theano
import theano.tensor as T
import numpy as np

import model_nnlm


def get_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_elapsed_time_min(start_seconds, end_seconds):
    return (end_seconds - start_seconds) / 60

def get_train_sample_num(i, batch_size, num_train_batches, num_train_samples):
    ngram_start_id = i * batch_size
    ngram_end_id = (i + 1) * batch_size - 1 if i < (num_train_batches - 1) else num_train_samples - 1
    return ngram_end_id - ngram_start_id + 1

def open_train_files(train_file, src_lang, tgt_lang, vocab_size):
    src_train_file = train_file + '.' + \
        str(vocab_size) + '.id.' + src_lang
    tgt_train_file = train_file + '.' + \
        str(vocab_size) + '.id.' + tgt_lang
    src_f = codecs.open(src_train_file, 'r', 'utf-8')
    tgt_f = codecs.open(tgt_train_file, 'r', 'utf-8')
    align_f = codecs.open(train_file + '.align', 'r', 'utf-8')
    return (src_f, tgt_f, align_f)

def print_cml_args(args):
    argsDict = vars(args)
    print "------------- PROGRAM PARAMETERS -------------"
    for a in argsDict:
        print "%s: %s" % (a, str(argsDict[a]))
    print "------------- ------------------ -------------"
    
class TrainModel(object):
    def __init__(self):
        self.vocab_loaded = False
        self.model_param_loaded = False
        self.train_param_loaded = False
        self.valid_set_loaded = False
        self.test_set_loaded = False

    def loadVocab(self, src_lang, tgt_lang, tgt_vocab_size, src_vocab_file, tgt_vocab_file):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tgt_vocab_size = tgt_vocab_size
        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file
        self.vocab_loaded = True

    def loadModelParams(self):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def loadTrainParams(self, train_file, batch_size, default_learning_rate, opt, valid_freq, finetune_epoch, 
        chunk_size=10000, vocab_size=1000, n_epochs=5):
        self.train_file = train_file
        self.batch_size = batch_size
        self.default_learning_rate = default_learning_rate
        self.opt = opt
        self.valid_freq = valid_freq
        self.finetune_epoch = finetune_epoch
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.n_epochs = n_epochs
        self.train_param_loaded = True

    def loadValidSet(self, valid_data_package):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def loadTestSet(self, test_data_package):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def validate(self, model, num_ngrams, batch_size, num_batches):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def test(self, model, num_ngrams, batch_size, num_batches):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    # Return False if there is no more data
    def loadBatchData(self, isInitialLoad=False):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def buildModels(self):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    # Open training files, called once in train function
    def openFiles(self):
        self.src_f, self.tgt_f, self.align_f = open_train_files(self.train_file, self.src_lang, self.tgt_lang, self.vocab_size)

    # Close training files, called once in train function
    def closeFiles(self):
        self.src_f.close()
        self.tgt_f.close()
        self.align_f.close()

    def trainOnBatch(self, train_model, i, batch_size, num_train_batches, num_train_samples, learning_rate):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def displayFirstNExamples(self, n):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")        

    def train(self):
        if not (self.vocab_loaded and self.model_param_loaded and self.train_param_loaded and self.valid_set_loaded):
            sys.stderr.write("TrainModel not initialized with enough parameters!\n")
            sys.exit(-1)

        # SGD HYPERPARAMETERS
        batch_size = self.batch_size
        learning_rate = self.default_learning_rate
        valid_freq = self.valid_freq
        finetune_epoch = self.finetune_epoch
        n_epochs = self.n_epochs
        model = self.model
        
        # Open files and load initial batch of data
        print "Getting initial data ..." 
        self.openFiles()
        self.loadBatchData(True)
        print "Display example input data ..."
        self.displayFirstNExamples(10)
        print ""
        
        ##### MODELING BEG #############################################
        train_model, valid_model, test_model = self.buildModels()
        ##### MODELING END #############################################

        ##### TRAINING BEG #############################################
        iter = 0
        epoch = 1
        start_iter = 0
        train_batches_epoch = 0
        train_costs = []
        best_valid_perp = float('inf')
        bestvalid_test_perp = 0
        prev_valid_loss = 0 # for learning rate halving
        epoch_valid_perps = [] # for convergence check
        seq_epoch_num = 5 # think model converges if perp not change in seq_epoch_num epochs
        # finetuning
        # the fraction of an epoch that we halve our learning rate
        finetune_fraction = 1
        assert finetune_epoch >= 1
        # printing info bookkeeping
        train_sample_per_epoch = 0
        train_sample_cnt = 0
        start_time_seconds = time.time()
        ##########################
        # epoch:
        # num_train_batch:
        ##########################
        while (epoch <= n_epochs):
            print "-------- EPOCH %d BEGINS ----------" % (epoch)
            while(True):
                num_train_samples = model.getTrainSetXSize()
                num_train_batches = (num_train_samples - 1) / batch_size + 1
                if epoch == 1:
                    train_batches_epoch += num_train_batches
                    train_sample_per_epoch += num_train_samples

                # train
                for i in xrange(num_train_batches):
                    outputs = self.trainOnBatch(train_model, i, batch_size, num_train_batches, num_train_samples, learning_rate)
                    train_sample_cnt += get_train_sample_num(i, batch_size, num_train_batches, num_train_samples)
                    train_costs.append(outputs[0])

                    # # check for nan/inf and print out debug infos
                    if np.isnan(outputs[0]) or np.isinf(outputs[0]):
                        sys.stderr.write('---> Epoch %d, iter %d: nan or inf, bad ... Stop training and exit\n' % (epoch, iter))
                        sys.exit(1)

                    iter += 1

                    # finetuning
                    if iter % (train_batches_epoch * finetune_fraction) == 0:
                        (valid_loss, valid_perp) = self.validate(valid_model, model.num_valid_ngrams, batch_size, model.num_valid_batches)
                        # if likelihood loss is worse than the previous epoch, the learning rate is multiplied by 0.5.
                        if epoch > finetune_epoch and valid_loss > prev_valid_loss:
                            learning_rate /= 2
                            print '---> Epoch %d, iter %d, halving learning rate to: %f' % (epoch, iter, learning_rate)
                        prev_valid_loss = valid_loss

                    if iter % valid_freq == 0:
                        train_loss = np.mean(train_costs)
                        (valid_loss, valid_perp) = self.validate(valid_model, model.num_valid_ngrams, batch_size, model.num_valid_batches)
                        (test_loss, test_perp) = self.test(test_model, model.num_test_ngrams, batch_size, model.num_test_batches)
                        elapsed_time_min = get_elapsed_time_min(start_time_seconds, time.time())
                        print 'iter: %d \t train_loss = %.4f \t valid_loss = %.4f \t test_loss = %.4f \t valid_perp = %.2f \t test_perp = %.2f \t time = %d min \t #samples: %d \t speed: %.0f samples/sec' % (iter, train_loss, valid_loss, test_loss, \
                            valid_perp, test_perp, elapsed_time_min, train_sample_cnt, train_sample_cnt/elapsed_time_min/60)
                        if valid_perp < best_valid_perp:
                            best_valid_perp = valid_perp
                            bestvalid_test_perp = test_perp

                # read more data
                if self.loadBatchData() ==  False:
                    break

            # end an epoch
            print "-------- EPOCH %d FINISHES --------" % (epoch)
            if iter > 1000:
                print "-------- (best valid perp = %.2f \t test_perp = %.2f) -----" % (best_valid_perp, bestvalid_test_perp)
                # Convergence check
                epoch_valid_perps.append(best_valid_perp)
                l = len(epoch_valid_perps)
                if l >= seq_epoch_num:
                    converged = True
                    for ii in range(seq_epoch_num):
                        if epoch_valid_perps[l-ii-1] != best_valid_perp:
                            converged = False
                else:
                    converged = False
                if converged:
                    print "-------- Converged --------"
                    break
            else:
                print "-------- Cannot give best valid perp (have not reached 1000 iterations) -----"
            if epoch == 1:
                print "-------- (%d samples/epoch) --------" % (train_sample_per_epoch)
            epoch = epoch + 1

            self.closeFiles()
            self.openFiles()
            self.loadBatchData()
        ##### TRAINING END #############################################
        print "--------"
        print "-------- Final Result: best valid perp = %.2f \t test perp = %.2f" % (best_valid_perp, bestvalid_test_perp)
        print "--------"
        self.closeFiles()

