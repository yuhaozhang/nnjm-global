#!/usr/bin/env python

"""
Util functions that are used to parse data into n-gram inputs.
Note that the data should NOT be the raw text file. The vocabulary should be extracted
and each text sentence should be mapped to a sequence of corresponding indeces in the
vocabulary.
"""

debug = False

import sys
import re
import codecs

import cPickle
import random
import numpy as np
import theano
reload(sys)
sys.setdefaultencoding("utf-8")

import io_vocab

def shared_dataset(data_package):
    shared_package = ()
    for d in data_package:
        shared_package += (theano.shared(
        np.asarray(d, dtype=theano.config.floatX), borrow=True), )
    return shared_package


def get_sentence_matrix_range(data_x, ngram_start_id, ngram_end_id):
    """
    This function assume that the input data_x is not shuffled.
    """
    sm_start_id = data_x[ngram_start_id][-1]
    sm_end_id = data_x[ngram_end_id-1][-1] + 1
    return (sm_start_id, sm_end_id)


def aggregate_alignments(align_line):
    """
    Parse the alignment file.

    Return:
      - s2t: a dict mapping source position to target position
      - t2s: a dict mapping target position to source position
    """
    align_tokens = re.split('\s+', align_line.strip())
    s2t = {}
    t2s = {}
    # process alignments
    for align_token in align_tokens:
        if align_token == '':
            continue
        (src_pos, tgt_pos) = re.split('\-', align_token)
        src_pos = int(src_pos)
        tgt_pos = int(tgt_pos)
        if src_pos not in s2t:
            s2t[src_pos] = []
        s2t[src_pos].append(tgt_pos)

        if tgt_pos not in t2s:
            t2s[tgt_pos] = []
        t2s[tgt_pos].append(src_pos)

    return (s2t, t2s)


def get_src_ngram(src_pos, src_ids_str, src_window, src_sos_index, src_eos_index, tgt_vocab_size):
    """
    Get a n-gram for a position in src sentence. The n-gram should be of length src_window, 
    and should center around the src_pos.
    """

    src_ngram = []
    src_len = len(src_ids_str)

    # NO SRC WORDS
    if src_window < 0:
        return src_ngram

    # left
    for ii in xrange(src_window):
        if (src_pos - src_window + ii) < 0:  # sos
            src_id = src_sos_index
        else:
            src_id = int(src_ids_str[src_pos - src_window + ii])
        src_ngram.append(src_id + tgt_vocab_size)

    # current word
    src_id = int(src_ids_str[src_pos])
    src_ngram.append(src_id + tgt_vocab_size)

    # right
    for ii in xrange(src_window):
        if (src_pos + ii + 1) >= src_len:  # eos
            src_id = src_eos_index
        else:
            src_id = int(src_ids_str[src_pos + ii + 1])
        src_ngram.append(src_id + tgt_vocab_size)

    return src_ngram


def get_src_pos(tgt_pos, t2s):
    """
    Get aligned src pos by average if there're multiple alignments. Return -1 if no alignment.
    This function should ONLY be used by infer_src_pos, when in global NNJM we need to infer
    a src position from a tgt position.
    """
    if tgt_pos in t2s:
        src_pos = 0
        for src_aligned_pos in t2s[tgt_pos]:
            src_pos += src_aligned_pos
        return int(src_pos / len(t2s[tgt_pos]))
    else:
        return -1


def infer_src_pos(tgt_pos, t2s, tgt_len):
    """
    Infer src aligned pos. Try to look around if there's no direct alignment
    """
    src_pos = get_src_pos(tgt_pos, t2s)
    if src_pos == -1:  # unaligned word, try to search alignments around
        k = 1
        while (tgt_pos - k) >= 0 or (tgt_pos + k) < tgt_len:
            if(tgt_pos - k) >= 0:  # left
                src_pos = get_src_pos(tgt_pos - k, t2s)
            if src_pos == -1 and (tgt_pos + k) < tgt_len:  # right
                src_pos = get_src_pos(tgt_pos + k, t2s)
            if src_pos != -1:
                break
            k += 1
            # if k>=3: break
    return src_pos


def print_joint_ngram(x, y, src_window, src_words, tgt_words, tgt_vocab_size):
    if src_window < 0:
        src_ngram_size = 0
    else:
        src_ngram_size = 2 * src_window + 1
    src_context = [src_words[x[j] - tgt_vocab_size]
                   for j in xrange(src_ngram_size)]
    context = [tgt_words[x[j]] for j in xrange(src_ngram_size, len(x))]
    sys.stderr.write('  %s ||| %s -> %s\n' %
                     (' '.join(src_context), ' '.join(context), tgt_words[y]))


def get_joint_ngrams(src_f, tgt_f, align_f, tgt_vocab_size, ngram_size, src_window, opt, num_read_lines=-1):
    """
    Return training example pairs in a NNJM model, given a triple of source file, target file and alignment file, and a few auxillary arguments.
    """
    src_sos_index = io_vocab.VocabConstants.SOS_INDEX
    tgt_sos_index = io_vocab.VocabConstants.SOS_INDEX
    src_eos_index = io_vocab.VocabConstants.EOS_INDEX
    tgt_eos_index = io_vocab.VocabConstants.EOS_INDEX

    x = []  # training examples
    y = []  # labels
    global debug
    num_ngrams = 0
    num_lines = 0

    predict_pos = (ngram_size - 1)  # predict last word
    if opt == 1:  # predict middle word
        predict_pos = (ngram_size - 1) / 2
    start_ngram = [tgt_sos_index for i in xrange(predict_pos)]
    end_ngram = [tgt_eos_index for i in xrange(ngram_size - predict_pos)]
    assert len(start_ngram) == predict_pos
    assert len(start_ngram) + len(end_ngram) == ngram_size

    for align_line in align_f:
        src_line = src_f.readline().strip()
        tgt_line = tgt_f.readline().strip()
        align_line = align_line.strip()
        src_ids_str = re.split('\s+', src_line)
        # skip examples that are too short
        if len(src_ids_str) <= 1:
            continue
        tgt_ids_str = re.split('\s+', tgt_line)
        (s2t, t2s) = aggregate_alignments(align_line)
        if len(src_ids_str) <= 2 or len(tgt_ids_str) <= 2:
            continue
        if len(t2s) == 0:
            continue

        tgt_orig_len = len(tgt_ids_str)
        if debug == True:
            sys.stderr.write('  src: %s\n' % src_line)
            sys.stderr.write('  tgt: %s\n' % tgt_line)
            sys.stderr.write('  align: %s\n' % align_line)
            sys.stderr.write('  s2t: %s\n' % str(s2t))
            sys.stderr.write('  t2s: %s\n' % str(t2s))
            sys.stderr.write('  tgt_orig_len: %d\n' % tgt_orig_len)
        if tgt_orig_len == 0:
            continue
        tgt_ids_str = start_ngram + tgt_ids_str + end_ngram

        # start (ngram_size-1)
        ngram = []
        for pos in xrange(ngram_size - 1):
            ngram.append(int(tgt_ids_str[pos]))

        # continue
        for pos in xrange(ngram_size - 1, len(tgt_ids_str)):
            ngram.append(int(tgt_ids_str[pos]))

            # get src ngram
            tgt_pos = pos - predict_pos  # predict_pos = len(start_ngram)
            src_pos = infer_src_pos(tgt_pos, t2s, tgt_orig_len)
            if src_pos == -1:
                continue

            # src part
            src_tgt_ngram = get_src_ngram(
                src_pos, src_ids_str, src_window, src_sos_index, src_eos_index, tgt_vocab_size)

            # tgt part
            for ii in xrange(ngram_size):
                if ii != predict_pos:
                    src_tgt_ngram.append(ngram[ii])
            x.append(src_tgt_ngram)
            y.append(ngram[predict_pos])
            num_ngrams += 1

            # remove prev word
            ngram.pop(0)

        num_lines += 1
        if num_lines == num_read_lines:
            break
    return (x, y)


def get_all_joint_ngrams(src_file, tgt_file, align_file, tgt_vocab_size, ngram_size, src_window, opt, num_read_lines=-1):
    """
    A wrapper around get_joint_ngrams function for handling file operations.
    """
    src_f = codecs.open(src_file, 'r', 'utf-8')
    tgt_f = codecs.open(tgt_file, 'r', 'utf-8')
    align_f = codecs.open(align_file, 'r', 'utf-8')
    # sys.stderr.write('# Loading ngrams from %s %s %s ...\n' %
    #                  (src_file, tgt_file, align_file))
    # sys.stderr.write(' tgt_vocab_size=%d, ngram_size=%d, src_window=%d\n' % (
    #     tgt_vocab_size, ngram_size, src_window))
    (x, y) = get_joint_ngrams(src_f, tgt_f, align_f, tgt_vocab_size, ngram_size, src_window, opt, num_read_lines)
    src_f.close()
    tgt_f.close()
    align_f.close()
    # sys.stderr.write('  num ngrams extracted=%d\n' % (len(y)))
    return (x, y)


def get_all_joint_ngrams_with_src_global(src_file, tgt_file, align_file, max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, num_read_lines=-1):
    src_f = codecs.open(src_file, 'r', 'utf-8')
    tgt_f = codecs.open(tgt_file, 'r', 'utf-8')
    align_f = codecs.open(align_file, 'r', 'utf-8')
    # sys.stderr.write('# Loading ngrams from %s %s %s ...\n' %
                     # (src_file, tgt_file, align_file))
    # sys.stderr.write('# tgt_vocab_size=%d, ngram_size=%d, src_window=%d\n' % (tgt_vocab_size, ngram_size, src_window))
    (x, y) = get_joint_ngrams_with_src_global(src_f, tgt_f, align_f, max_src_sent_length, \
        tgt_vocab_size, ngram_size, src_window, opt, num_read_lines)
    src_f.close()
    tgt_f.close()
    align_f.close()
    return (x, y)


def get_joint_ngrams_with_src_global(src_f, tgt_f, align_f, max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, num_read_lines=-1):
    src_sos_index = io_vocab.VocabConstants.SOS_INDEX
    tgt_sos_index = io_vocab.VocabConstants.SOS_INDEX
    src_eos_index = io_vocab.VocabConstants.EOS_INDEX
    tgt_eos_index = io_vocab.VocabConstants.EOS_INDEX
    
    x = []  # training examples
    y = []  # labels
    global debug
    num_ngrams = 0
    num_lines = 0

    predict_pos = (ngram_size - 1)  # predict last word
    if opt == 1:  # predict middle word
        predict_pos = (ngram_size - 1) / 2
    start_ngram = [tgt_sos_index for i in xrange(predict_pos)]
    end_ngram = [tgt_eos_index for i in xrange(ngram_size - predict_pos)]
    assert len(start_ngram) == predict_pos
    assert len(start_ngram) + len(end_ngram) == ngram_size

    # l = 0
    for align_line in align_f:
        src_line = src_f.readline().strip()
        tgt_line = tgt_f.readline().strip()
        align_line = align_line.strip()
        src_ids_str = re.split('\s+', src_line)
        # skip examples that are too short
        if len(src_ids_str) <= 1:
            continue
        tgt_ids_str = re.split('\s+', tgt_line)
        # add src sentence vector
        sent_vector = []
        for ii in xrange(len(src_ids_str)):
            sent_vector.append(int(src_ids_str[ii]) + tgt_vocab_size)
        # use </s> to fill the vector to force all vectors to be equal length
        for ii in xrange(max_src_sent_length - len(src_ids_str)):
            sent_vector.append(src_eos_index + tgt_vocab_size)

        (s2t, t2s) = aggregate_alignments(align_line)
        if len(src_ids_str) <= 2 or len(tgt_ids_str) <= 2:
            continue
        if len(t2s) == 0:
            continue

        tgt_orig_len = len(tgt_ids_str)
        if debug == True:
            sys.stderr.write('  src: %s\n' % src_line)
            sys.stderr.write('  tgt: %s\n' % tgt_line)
            sys.stderr.write('  align: %s\n' % align_line)
            sys.stderr.write('  s2t: %s\n' % str(s2t))
            sys.stderr.write('  t2s: %s\n' % str(t2s))
            sys.stderr.write('  tgt_orig_len: %d\n' % tgt_orig_len)
        if tgt_orig_len == 0:
            continue
        tgt_ids_str = start_ngram + tgt_ids_str + end_ngram

        # start (ngram_size-1)
        ngram = []
        for pos in xrange(ngram_size - 1):
            ngram.append(int(tgt_ids_str[pos]))

        # continue
        for pos in xrange(ngram_size - 1, len(tgt_ids_str)):
            ngram.append(int(tgt_ids_str[pos]))

            # get src ngram
            tgt_pos = pos - predict_pos  # predict_pos = len(start_ngram)
            src_pos = infer_src_pos(tgt_pos, t2s, tgt_orig_len)
            if src_pos == -1:
                continue

            # src part
            src_tgt_ngram = get_src_ngram(
                src_pos, src_ids_str, src_window, src_sos_index, src_eos_index, tgt_vocab_size)

            # tgt part
            for ii in xrange(ngram_size):
                if ii != predict_pos:
                    src_tgt_ngram.append(ngram[ii])

            x.append(src_tgt_ngram + sent_vector)
            y.append(ngram[predict_pos])
            num_ngrams += 1

            # remove prev word
            ngram.pop(0)

        num_lines += 1
        if num_lines == num_read_lines:
            break

    return (x, y)


def get_all_joint_ngrams_with_src_global_matrix(src_file, tgt_file, align_file, max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, 
    num_read_lines=-1, stopword_cutoff=-1):
    src_f = codecs.open(src_file, 'r', 'utf-8')
    tgt_f = codecs.open(tgt_file, 'r', 'utf-8')
    align_f = codecs.open(align_file, 'r', 'utf-8')
    # sys.stderr.write('# Loading ngrams from %s %s %s ...\n' %
                     # (src_file, tgt_file, align_file))
    # sys.stderr.write('# tgt_vocab_size=%d, ngram_size=%d, src_window=%d\n' % (tgt_vocab_size, ngram_size, src_window))
    (x, y, sentence_matrix) = get_joint_ngrams_with_src_global_matrix(src_f, tgt_f, align_f, max_src_sent_length, \
        tgt_vocab_size, ngram_size, src_window, opt, num_read_lines, stopword_cutoff)
    src_f.close()
    tgt_f.close()
    align_f.close()
    return (x, y, sentence_matrix)

def get_joint_ngrams_with_src_global_matrix(src_f, tgt_f, align_f, max_src_sent_length, tgt_vocab_size, ngram_size, src_window, opt, 
    num_read_lines=-1, stopword_cutoff=-1):
    """
    No shuffle for this function.
    """

    src_sos_index = io_vocab.VocabConstants.SOS_INDEX
    tgt_sos_index = io_vocab.VocabConstants.SOS_INDEX
    src_eos_index = io_vocab.VocabConstants.EOS_INDEX
    tgt_eos_index = io_vocab.VocabConstants.EOS_INDEX
    
    x = []  # training examples
    y = []  # labels
    sentence_matrix = []
    global debug
    num_ngrams = 0
    num_lines = 0

    predict_pos = (ngram_size - 1)  # predict last word
    if opt == 1:  # predict middle word
        predict_pos = (ngram_size - 1) / 2
    start_ngram = [tgt_sos_index for i in xrange(predict_pos)]
    end_ngram = [tgt_eos_index for i in xrange(ngram_size - predict_pos)]
    assert len(start_ngram) == predict_pos
    assert len(start_ngram) + len(end_ngram) == ngram_size

    # l = 0
    for align_line in align_f:
        src_line = src_f.readline().strip()
        tgt_line = tgt_f.readline().strip()
        align_line = align_line.strip()
        src_ids_str = re.split('\s+', src_line)
        tgt_ids_str = re.split('\s+', tgt_line)

        if len(src_ids_str) <= 1:
            continue
        
        # create src sentence vector (first item in this vector is the sentence length, without </s>)
        sent_vector = []
        sent_len = 0
        for ii in xrange(len(src_ids_str)):
            src_token_id = int(src_ids_str[ii])
            if stopword_cutoff > 0 and src_token_id < stopword_cutoff:
                continue
            sent_vector.append(src_token_id + tgt_vocab_size)
            sent_len += 1
        # Add sentence length to the first item
        if sent_len == 0:
            sent_len = 1 # if the length is 0, then use a single </s> to represent the sentence
        sent_vector.insert(0, sent_len)
        # use </s> to fill the vector to force all vectors to be equal length
        real_sent_vector_len = len(sent_vector) - 1
        for ii in xrange(max_src_sent_length - real_sent_vector_len):
            sent_vector.append(src_eos_index + tgt_vocab_size)

        # add sentence vector into sentence matrix
        sentence_matrix.append(sent_vector)

        (s2t, t2s) = aggregate_alignments(align_line)
        if len(src_ids_str) <= 2 or len(tgt_ids_str) <= 2:
            continue
        if len(t2s) == 0:
            continue

        tgt_orig_len = len(tgt_ids_str)
        if debug == True:
            sys.stderr.write('  src: %s\n' % src_line)
            sys.stderr.write('  tgt: %s\n' % tgt_line)
            sys.stderr.write('  align: %s\n' % align_line)
            sys.stderr.write('  s2t: %s\n' % str(s2t))
            sys.stderr.write('  t2s: %s\n' % str(t2s))
            sys.stderr.write('  tgt_orig_len: %d\n' % tgt_orig_len)
        if tgt_orig_len == 0:
            continue
        tgt_ids_str = start_ngram + tgt_ids_str + end_ngram

        # start (ngram_size-1)
        ngram = []
        for pos in xrange(ngram_size - 1):
            ngram.append(int(tgt_ids_str[pos]))

        # continue
        for pos in xrange(ngram_size - 1, len(tgt_ids_str)):
            ngram.append(int(tgt_ids_str[pos]))

            # get src ngram
            tgt_pos = pos - predict_pos  # predict_pos = len(start_ngram)
            src_pos = infer_src_pos(tgt_pos, t2s, tgt_orig_len)
            if src_pos == -1:
                continue

            # src part
            src_tgt_ngram = get_src_ngram(
                src_pos, src_ids_str, src_window, src_sos_index, src_eos_index, tgt_vocab_size)

            # tgt part
            for ii in xrange(ngram_size):
                if ii != predict_pos:
                    src_tgt_ngram.append(ngram[ii])

            # append the sentence vector index (= num_lines) into the src_tgt_ngram vector
            src_tgt_ngram.append(num_lines)

            x.append(src_tgt_ngram)
            y.append(ngram[predict_pos])
            num_ngrams += 1

            # remove prev word
            ngram.pop(0)

        num_lines += 1
        if num_lines == num_read_lines:
            break

    return (x, y, sentence_matrix)
