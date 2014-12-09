#!/usr/bin/env python

"""
Functions that extract vocab from the text data.
"""
import os
import sys
import re
import codecs

class VocabConstants():
    """
    A wrapper of a few constants used in NNLM and NNJM vocabs.
    """
    UNK = "<unk>"
    SOS = "<s>"
    EOS = "</s>"
    UNK_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2

def load_text(in_file):
    """
    Load text from a file using utf-8.

    Return:
      - lines: a list of all lines after striped
    """
    f = codecs.open(in_file, 'r', 'utf-8')
    lines = []
    for line in f:
        lines.append(line.strip())
    f.close()
    return lines

def get_vocab(corpus_file, vocab_file, freq, vocab_size):
    """
    Handle vocab file in a pipeline: if vocab fie exists, simply load it; if it does not exist,
    compute it from corpus, write to file and return it.

    Return:
      - words: all words in vocab as string
      - vocab_map: string to int mapping of words
      - vocab_size: size of vocab, should be equal to len(words)
    """
    if os.path.isfile(vocab_file):  # vocab_file exist
        (vocab_map, vocab_size) = load_vocab(vocab_file)
    else:
        (words, vocab_map, freq_map, vocab_size, num_train_words,
         num_lines) = load_vocab_from_corpus(corpus_file, freq, vocab_size)
        write_vocab(vocab_file, words)
    return (vocab_map, vocab_size)


def get_mapped_sentence(corpus_file, vocab_map, sentence_map_file):
    """
    Convert each sentence into integer list according to the vocab.

    Return:
      - sentences: a list of list, where a secondary list correponds to a id sequence of a sentence
    """
    if os.path.isfile(sentence_map_file):
        sentences = load_mapped_sentence(sentence_map_file)
    else:
        sentences = []
        lines = load_text(corpus_file)
        for line in lines:
            tokens = line.strip().split()
            mapping = to_id_int(tokens, vocab_map, 0)
            sentences.append(mapping)
        write_mapped_sentence(sentences, sentence_map_file)
    return sentences


def write_mapped_sentence(sentences, sentence_map_file):
    f = open(sentence_map_file, 'w')
    for sentence in sentences:
        sentence = [str(x) for x in sentence]
        f.write(' '.join(sentence) + '\n')
    return


def load_mapped_sentence(sentence_map_file):
    sentences = []
    lines = load_text(sentence_map_file)
    for line in lines:
        ids = line.strip().split()
        int_ids = [int(i) for i in ids]
        sentences.append(int_ids)
    return sentences


def add_word_to_vocab(word, words, vocab_map, vocab_size):
    """
    Add a word to a existing vocab. If word already exists, do nothing.
    Especially useful for handling <s>, </s> and <unk>.

    Return: updated vocab.
    """
    if word not in vocab_map:
        words.append(word)
        vocab_map[word] = vocab_size
        vocab_size += 1
        #sys.stderr.write('  add %s\n' % word)
    return (words, vocab_map, vocab_size)


def to_id_int(tokens, vocab_map, offset=0):
    """
    Map a list of tokens to their correponding int representations (id).

    Return: list of ids in the same order as input tokens.
    """
    unk = VocabConstants.UNK
    return [int(vocab_map[token] + offset) if token in vocab_map else str(vocab_map[unk] + offset) for token in tokens]


def to_id(tokens, vocab_map, offset=0):
    """
    Map a list of tokens to their correponding int representations (id).

    Return: list of ids in the same order as input tokens.
    """
    unk = VocabConstants.UNK
    return [str(vocab_map[token] + offset) if token in vocab_map else str(vocab_map[unk] + offset) for token in tokens]


def to_text(indices, words, offset=0):
    """
    Map a list of ids to string words.

    Return: a list of string words as in input indices.
    """
    return [words[int(index) - offset] for index in indices]


def write_vocab(out_file, words, freqs=[]):
    """
    Write vocab to a file. If a frequency list (freqs) is provided, write as "word freq";
    if nothing is provided, simply write the word.
    """
    f = codecs.open(out_file, 'w', 'utf-8')
    # sys.stderr.write('# Output vocab to %s ...\n' % out_file)
    vocab_size = 0
    for word in words:
        #f.write('%s %d\n' % (word, vocab_size))
        if len(freqs) == 0:
            f.write('%s\n' % word)
        else:
            f.write('%s %d\n' % (word, freqs[vocab_size]))
        vocab_size += 1
    f.close()
    # sys.stderr.write('  num words = %d\n' % vocab_size)


def load_vocab(in_file):
    """
    Load vocab from a vocab file.

    Return as get_vocab().
    """
    sos = VocabConstants.SOS
    eos = VocabConstants.EOS
    unk = VocabConstants.UNK

    # sys.stderr.write('# Loading vocab file %s ...\n' % in_file)
    vocab_inf = codecs.open(in_file, 'r', 'utf-8')
    words = []
    vocab_map = {}
    vocab_size = 0
    # add word, and update vocab_map based on the size count
    for line in vocab_inf:
        tokens = re.split('\s+', line.strip())
        word = tokens[0]
        words.append(word)
        vocab_map[word] = vocab_size
        vocab_size += 1

    # add sos, eos, unk
    for word in [sos, eos, unk]:
        (words, vocab_map, vocab_size) = add_word_to_vocab(
            word, words, vocab_map, vocab_size)
    vocab_inf.close()
    # sys.stderr.write('  num words = %d\n' % vocab_size)
    return (vocab_map, vocab_size)

def inverse_vocab(vocab):
    """
    Get inverse vocab, namely a word list from a vocab.

    Return as get_vocab().
    """
    inv_vocab = {}
    for k,v in vocab.iteritems():
        inv_vocab[v] = k
    return inv_vocab

def load_vocab_from_corpus(in_file, freq, max_vocab_size):
    """
    Load vocab information from corpus, and maintain vocab mapping and freq mapping.
    freq: frequency restriction can be provided, and rare words will be <unk>.
    max_vocab_size: Or instead max size could be provided to cut off the vocab.

    Return vocab information and basic corpus information.
    """
    f = codecs.open(in_file, 'r', 'utf-8')
    # sys.stderr.write('# Loading vocab from %s ... ' % in_file)

    words = []
    vocab_map = {}
    freq_map = {}
    vocab_size = 0
    num_train_words = 0
    num_lines = 0
    for line in f:
        tokens = re.split('\s+', line.strip())
        num_train_words += len(tokens)
        for token in tokens:
            if token not in vocab_map:
                words.append(token)
                vocab_map[token] = vocab_size
                freq_map[token] = 0
                vocab_size += 1
            freq_map[token] += 1

        num_lines += 1
        if num_lines % 100000 == 0:
            sys.stderr.write(' (%d) ' % num_lines)
            # break
    f.close()
    # sys.stderr.write('\n  vocab_size=%d, num_train_words=%d, num_lines=%d\n' % (
    #     vocab_size, num_train_words, num_lines))

    # if restrictions are provided, filter out rare words
    if freq > 0 or max_vocab_size > 0:
        (words, vocab_map, freq_map, vocab_size) = update_vocab(
            words, vocab_map, freq_map, freq, max_vocab_size)
    return (words, vocab_map, freq_map, vocab_size, num_train_words, num_lines)


def update_vocab(words, vocab_map, freq_map, freq, max_vocab_size):
    """
    Filter out rare words (<freq) or keep the top vocab_size frequent words
    """
    unk = VocabConstants.UNK
    sos = VocabConstants.SOS
    eos = VocabConstants.EOS
    new_words = [unk, sos, eos]
    new_vocab_map = {unk: VocabConstants.UNK_INDEX, sos: VocabConstants.SOS_INDEX, eos: VocabConstants.EOS_INDEX}
    new_freq_map = {unk: 0, sos: 0, eos: 0}
    vocab_size = 3
    if freq > 0:
        for word in words:
            if freq_map[word] < freq:  # rare
                new_freq_map[unk] += freq_map[word]
            else:
                new_words.append(word)
                new_vocab_map[word] = vocab_size
                new_freq_map[word] = freq_map[word]
                vocab_size += 1
        sys.stderr.write('  convert rare words (freq<%d) to %s: new vocab size=%d, unk freq=%d\n' % (
            freq, unk, vocab_size, new_freq_map[unk]))
    else:
        assert(max_vocab_size > 0)
        sorted_items = sorted(
            freq_map.items(), key=lambda x: x[1], reverse=True)
        for (word, freq) in sorted_items:
            new_words.append(word)
            new_vocab_map[word] = vocab_size
            new_freq_map[word] = freq
            vocab_size += 1
            if vocab_size == max_vocab_size:
                break
        # sys.stderr.write('  update vocab: new vocab size=%d\n' % (vocab_size))
    return (new_words, new_vocab_map, new_freq_map, vocab_size)

def getOriginalIndeces(indeces, offset):
    orig = []
    for i in indeces:
        orig.append(i - offset)
    return orig

def getWordsFromIndeces(indeces, vocab, offset):
    words = []
    for i in indeces:
        words.append(vocab[i - offset])
    return words
