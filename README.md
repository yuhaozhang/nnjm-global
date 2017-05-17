Neural Network Joint Language Model (NNJM) using Python/Theano
==========

**Update**: You can find the **technical report** corresponding to this code base here: [Neural Network Joint Language Model: An Investigation and An Extension With Global Source Context](http://yuhao.im/files/ZhangQi_NNJM.pdf), by Yuhao Zhang and Charles R. Qi.

This is an implementation of a neural network joint language model (NNJM) using Python. NNJM is proposed in the context of machine translation and jointly model target language and its aligned source language to improve machine translation performance. The implementation presented here is based on a paper authored by BBN: [Fast and Robust Neural Network Joint Models for Statistical Machine
Translation](http://acl2014.org/acl2014/P14-1/pdf/P14-1129.pdf). Besides, this implementation also allows for an extension of the NNJM model, which utilizes the entire source sentence (global source context) to improve the model peroformance measured by perplexity. We called this NNJM-Global for simplicity. Thus, this code could run in three modes:

- **NNLM**: model only the target language with target N-gram.
- **NNJM**: jointly model the target language and source language with target N-gram and source window.
- **NNJM-Global**: jointly model the target language and source language with target N-gram, source window, and entire source sentence.

Built on top of Theano, our code could also be run on GPU supported by CUDA with no additional effort. GPU computation can speed up the neural network training for ~x10 in many cases. Please see below for detail.

**Note**: The code base and documentation is still in preparation. You can check later for a cleaner version.

## Files

- README      - this file
- code/           - directory contains all the python code files.
- data/: each subdirectory contains 3 files -.(en|zh|align) where -.align contains alignments for a pair of sentences per line. Each line is a series of pairs Chinese positions - English positions.
	- bitex/        - toy training data
	- tune/         - tuning data
	- p1r6_dev/     - test data

## Dependency

This code is implemented in Python. For numeric computation, we use [**Numpy**](http://www.numpy.org/). For symbolic computation and the construction of the neural network, we use [**Theano**](http://deeplearning.net/software/theano/). Note that **Theano** is still in development, so the current version might be unstable sometimes.

## Usage

#### Data Preprocessing

You have to preprocess the original text data in order to run the following code. Please run under the root directory.

	python code/io_preprocess.py --joint --src_lang en --tgt_lang zh data/bitex/tiny_training 1000 toy_joint
	python code/io_preprocess.py --joint --src_lang en --tgt_lang zh data/p1r6_dev/p1r6_dev 1000 toy_joint
	python code/io_preprocess.py --joint --src_lang en --tgt_lang zh data/tune/tiny_tune 1000 toy_joint

#### Train basic NNJM

This is an example on the toy dataset. Note that if you want to use GPU, you have to add the prefix `THEANO_FLAGS` setting. If you run it with CPU, you do not need to use the prefix.

	THEANO_FLAGS='device=gpu,floatX=float32' python ./code/train_nnjm_gplus.py --act_func tanh --learning_rate 0.5 --emb_dim 16 --hidden_layers 64 --src_lang zh --tgt_lang en ./data/bitex/tiny_training ./data/tune/tiny_tune ./data/p1r6_dev/p1r6_dev 5 100 1000 2 ./toy_joint

#### Train NNJM-Global

	THEANO_FLAGS='device=gpu,floatX=float32' python ./code/train_nnjm_global.py --act_func tanh --learning_rate 0.5 --emb_dim 16 --hidden_layers 64 --src_lang zh --tgt_lang en ./data/bitex/tiny_training ./data/tune/tiny_tune ./data/p1r6_dev/p1r6_dev 5 100 1000 ./toy_joint

## Acknowledgement

This code is based on an original implementation provided by Thang Luong in the Stanford Natural Language Processing Group.

## Authors

Thang Luong, Yuhao Zhang, Charles Ruizhongtai Qi @ Stanford University
