# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nltk.tokenize import word_tokenize

import collections
import os
import sys

import tensorflow as tf
import numpy as np

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      #return f.read().replace("\n", "<eos>").split()
      tokenized = ["eos"] + word_tokenize(f.read().replace("\n", " eos "))
      for i in range(len(tokenized)):
        if tokenized[i]=="eos":
          tokenized[i]=="<eos>"
        elif tokenized[i]=="unk":
          tokenized[i]=="<unk>"
      return tokenized
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab2(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      lines = f.readlines()
      print("Lines read")
      word_to_id = {}
      num_words = 0
      line_num = 0
      for line in lines:
        if line_num % 100000 == 0:
          print("in line", line_num)
        words_line = word_tokenize(line.replace("\n", " eos "))
        for w in words_line:
          if w not in word_to_id.keys():
            word_to_id[w] = num_words
            num_words += 1
        line_num += 1
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()

  print("words: ", num_words)
  return word_to_id

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  print("words: ", len(words))
  #print(words)
  return word_to_id

def _file_to_word_ids2(filename, word_to_id):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      lines = f.readlines()
      print("File to word ids in ", filename)
      data_ids = []
      line_num = 0
      for line in lines:
        if line_num % 100 == 0:
          print("in line", line_num)
        words_line = word_tokenize(line.replace("\n", " eos "))
        for w in words_line:
          if w in word_to_id.keys():
            data_ids.append(word_to_id[w])
          else:
            data_ids.append(word_to_id["unk"])
        line_num += 1
      return np.array(data_ids)
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return np.array([word_to_id[word] for word in data if word in word_to_id])

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  #train_path = os.path.join(data_path, "news30K.en.processed")
  #valid_path = os.path.join(data_path, "newstest2013.en.processed")
  #test_path = os.path.join(data_path, "newstest2013.en.processed")

  train_path = os.path.join(data_path, "news1M.2012.en.processed")
  valid_path = os.path.join(data_path, "val.small.small.en.processed")
  test_path = os.path.join(data_path, "val.small.small.en.processed")

  word_to_id = _build_vocab2(train_path)
  train_data = _file_to_word_ids2(train_path, word_to_id)
  valid_data = _file_to_word_ids2(valid_path, word_to_id)
  test_data = _file_to_word_ids2(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
