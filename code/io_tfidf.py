#!/usr/bin/env python

"""
A module that is used to compute the tfidf weight given a training corpus and a vocab.

Example input:
  - corpus filename = new_data/new_train
  - lang = fr
  - vocab_file_prefix = new_data
  - vocab_size = 20000

Example output:
  - new_data.20000.tfidf.fr

The vocab file must exist before run this program.

"""

usage = "A module that is used to compute the tfidf weight given a training corpus and a vocab."

import os
import sys
import time
import re
import codecs
import argparse
import datetime

# our libs
import io_vocab

def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """
  
  parser = argparse.ArgumentParser(description=usage) # add description
  # positional arguments
  parser.add_argument('filename', metavar='filename', type=str, help='the input file name (without extension)') 
  parser.add_argument('lang', metavar='lang', type=str, help='language suffix') 
  parser.add_argument('vocab_file_prefix', metavar='vocab_file_prefix', type=str, help='the prefix of vocab file') 
  parser.add_argument('vocab_size', metavar='vocab_size', type=int, help='the expected vocab size') 

  args = parser.parse_args()
  return args

def initialize_tfidf(vocab_map):
  vocab_list = vocab_map.values()
  tf_map = {}
  df_map = {}
  for w in vocab_list:
    tf_map[w] = 0
    df_map[w] = 0
  return (tf_map, df_map)

def update_tfidf(line, tf_map, df_map):
  pass

def compute_tfidf(tf_map, df_map):
  pass

def get_tfidf(input_file, vocab_file):
  (vocab_map, vocab_size) = io_vocab.load_vocab(vocab_file)
  print vocab_map
  exit()
  tf_map, df_map = initialize_tfidf(vocab_map)
  infile = open(input_file, 'r')
  for line in infile:
    update_tfidf(line, tf_map, df_map)
  tfidf_map = compute_tfidf(tf_map, df_map)

def main():
  args = process_command_line()
  lang = args.lang
  filename = args.filename
  vocab_file_prefix = args.vocab_file_prefix
  vocab_size = args.vocab_size
  print "-- Computing tfidf for file: %s, for language: %s, with vocab size: %d" % (filename, lang, vocab_size)

  input_file = filename + '.' + str(vocab_size) + '.id.' + lang
  vocab_file = vocab_file_prefix + '.' + str(vocab_size) + '.vocab.' + lang
  get_tfidf(input_file, vocab_file)


if __name__ == '__main__':
  main()
