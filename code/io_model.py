#!/usr/bin/env python

"""
A module that is used to save and load the model parameters.
"""

usage = "A module that is used to save and load the model parameters."

import os
import sys
import time
import re
import codecs
import cPickle


def save_model(model_file, classifier):
  sys.stderr.write('---> Save model to %s\n' % (model_file)) 
  save_file = open(model_file, 'wb')
  cPickle.dump(classifier.context_size, save_file, -1) # context_size
  cPickle.dump(classifier.num_hidden_layers, save_file, -1) # num hidden layers
  cPickle.dump(classifier.linearLayer.W_emb.get_value(), save_file, -1) # embeddings
  
  # hidden layers
  for ii in xrange(classifier.num_hidden_layers):
    cPickle.dump(classifier.hidden_layers[ii].W.get_value(), save_file, -1)
    cPickle.dump(classifier.hidden_layers[ii].b.get_value(), save_file, -1)
  
  # softmax
  cPickle.dump(classifier.softmaxLayer.W.get_value(), save_file, -1)
  cPickle.dump(classifier.softmaxLayer.b.get_value(), save_file, -1)
  save_file.close()

def print_matrix_stat(W, label):
  if W.ndim==1:
    num_rows = W.shape[0]
    num_cols = 1
  else:
    (num_rows, num_cols) = W.shape
  sys.stderr.write('%s [%d, %d]: min=%g, max=%g, avg=%g\n' % (label, num_rows, num_cols, W.min(), W.max(), W.mean()))

def load_model(model_file):
  sys.stderr.write('---> Loading model from %s ...\n' % model_file)
  f = file(model_file, 'rb')
  context_size = cPickle.load(f)
  num_hidden_layers = cPickle.load(f)
  sys.stderr.write('    -> context_size=%d\n' % context_size)
  sys.stderr.write('    -> num_hidden_layers=%d\n' % num_hidden_layers)
  
  linear_W_emb = cPickle.load(f)
  print_matrix_stat(linear_W_emb, '    -> W_emb')

  hidden_Ws = []
  hidden_bs = []
  for ii in xrange(num_hidden_layers):
    hidden_W = cPickle.load(f)
    hidden_b = cPickle.load(f)
    hidden_Ws.append(hidden_W)
    hidden_bs.append(hidden_b)
    print_matrix_stat(hidden_W, '    -> hidden_W_' + str(ii))
    print_matrix_stat(hidden_b, '    -> hidden_b_' + str(ii))

  softmax_W = cPickle.load(f)
  softmax_b = cPickle.load(f)
  print_matrix_stat(softmax_W, '    -> softmax_W ')
  print_matrix_stat(softmax_b, '    -> softmax_b ')
  f.close()

  return (context_size, linear_W_emb, hidden_Ws, hidden_bs, softmax_W, softmax_b)
