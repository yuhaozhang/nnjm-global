#!/usr/bin/env python

"""
A module that is used to map a io_vocab format data file into a integer file. The entire workflow consists of:
  1. First scan the file to extract vocabulary, according to a given vocab size; or read vocabulary from file if it already exists.
  2. Write the vocabulary to file if vocabulary not exists.
  3. Scan the file again: readin a sentence line, convert it into integer sequences, and write to output file.

Example input:
  - filename = tiny_training
  - vocab_size = 1000
  - is_joint = true
  - src_lang = en
  - tgt_lang = zh
  - vocab_file_prefix

Example output:
  - tiny_training.1000.id.en
  - tiny_training.1000.id.zh
  - tiny_training.1000.vocab.en
  - tiny_training.1000.vocab.zh

If the vocab_file exists, then it should use that file instead of creating new vocabs.

"""

usage = "A module that is used to map a io_vocab format data file into a integer file."

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
  parser.add_argument('vocab_size', metavar='vocab_size', type=int, help='the expected vocab size') 
  parser.add_argument('vocab_file_prefix', metavar='vocab_file_prefix', type=str, help='the prefix of vocab file') 

  # joint model
  parser.add_argument('--joint', dest='is_joint', action='store_true', default=False, help='to enable map joint model data file, where 2 languages are handled')
  parser.add_argument('--src_lang', dest='src_lang', type=str, default='', help='src lang (default=\'\')')
  parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='', help='tgt_lang (default=\'\')')

  args = parser.parse_args()
  return args

def map_file(filename, lang, vocab_size, vocab_file_prefix):
  """
  Extract vocab, and convert file into integer sequences, and write to output file.
  """
  input_file = filename + '.' + lang
  vocab_file = vocab_file_prefix + '.' + str(vocab_size) + '.vocab.' + lang
  (vocab_map, v) = io_vocab.get_vocab(input_file, vocab_file, -1, vocab_size)
  print "# vocab size is %d" % v
  output_file = filename + '.' + str(vocab_size) + '.id.' + lang
  sentences = io_vocab.get_mapped_sentence(input_file, vocab_map, output_file)
  max_len = max([len(x) for x in sentences])
  print "# max length of sentence is %d" % max_len

def main():
  args = process_command_line()
  is_joint = args.is_joint
  if is_joint == False:
    print "-- Mapping file: %s, for language: %s, with vocab size: %d" % (args.filename, 'en', args.vocab_size)
    map_file(args.filename, 'en', args.vocab_size, args.vocab_file_prefix)
  else:
    print "-- Mapping joint file: %s, for language: %s and %s, with vocab size: %d" % (args.filename, args.src_lang, args.tgt_lang, args.vocab_size)
    map_file(args.filename, args.src_lang, args.vocab_size, args.vocab_file_prefix)
    map_file(args.filename, args.tgt_lang, args.vocab_size, args.vocab_file_prefix)

if __name__ == '__main__':
  main()
