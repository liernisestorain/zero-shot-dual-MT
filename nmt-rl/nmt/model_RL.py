# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
import numpy as np
import pickle

from tensorflow.python.layers import core as layers_core

from . import ptb_word_lm
from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
import subprocess
import os

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.
    """

    assert isinstance(iterator, iterator_utils.BatchedInput)
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.num_layers = hparams.num_layers
    self.num_gpus = hparams.num_gpus
    self.time_major = hparams.time_major
    ### Backtranslation parameters
    self.backtranslation = False
    self.sampling_lengths = None
    self.eos_id = hparams.eos
    
    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Initializer
    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    # TODO(ebrevdo): Only do this if the mode is TRAIN?
    self.init_embeddings(hparams, scope)
    self.batch_size = tf.size(self.iterator.source_sequence_length)

    # Projection
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer = layers_core.Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")

    ## Train graph
    res = self.build_graph(hparams, scope=scope)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.final_context_state, self.sample_id, _, _, _, _, _ = res
      #self.infer_logits, _, self.final_context_state, self.sample_id = res
      self.sample_words = reverse_target_vocab_table.lookup(
          tf.to_int64(self.sample_id))

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

    self.global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()

    ## Compute backtranslation
    if self.mode != tf.contrib.learn.ModeKeys.INFER:
        sampled_ids = res[4]
        sampling_lengths = res[5]
        source_sent = res[7]

        if self.time_major:
            sampled_ids = tf.transpose(sampled_ids, [1, 0])

        # Get tags
        en_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<en>")), tf.int32)
        es_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<es>")), tf.int32)
        fr_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<fr>")), tf.int32)
        unk_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<unk>")), tf.int32)
        sos_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<s>")), tf.int32)

        shape = tf.ones([self.batch_size], dtype=tf.int32)

        ## Prepare data for the language models
        # Prepend <sos> in the beginning of the sample sentences
        sos_tags = tf.multiply(shape, sos_tag)
        lm_input = tf.reshape(tf.transpose(sampled_ids), [-1])
        lm_input = tf.concat([sos_tags, lm_input], 0)
        lm_input = tf.transpose(tf.reshape(lm_input, [-1, self.batch_size]))

        # Replace language tags with the <unk> tag since the LM has never seen these tags
        ones = tf.ones_like(lm_input)
        lang_tags_en = tf.cast(tf.equal(lm_input, tf.multiply(ones, en_tag)), tf.int32)
        delta_en = en_tag - unk_tag
        lang_tags_en = tf.multiply(lang_tags_en, delta_en)
        lang_tags_es = tf.cast(tf.equal(lm_input, tf.multiply(ones, es_tag)), tf.int32)
        delta_es = es_tag - unk_tag
        lang_tags_es = tf.multiply(lang_tags_es, delta_es)
        lang_tags_fr = tf.cast(tf.equal(lm_input, tf.multiply(ones, fr_tag)), tf.int32)
        delta_fr = fr_tag - unk_tag
        lang_tags_fr = tf.multiply(lang_tags_fr, delta_fr)
        lm_input = tf.subtract(lm_input, lang_tags_en)
        lm_input = tf.subtract(lm_input, lang_tags_es)
        lm_input = tf.subtract(lm_input, lang_tags_fr)

        lm_lengths = tf.add(sampling_lengths, 1)

        with tf.variable_scope("LM_es"):
            lm_rewards_es = tf.stop_gradient(self.get_rewards(lm_input, lm_lengths))
        with tf.variable_scope("LM_fr"):
            lm_rewards_fr = tf.stop_gradient(self.get_rewards(lm_input, lm_lengths))
        source_tags = source_sent[0]
        mask_es = tf.cast(tf.equal(source_tags, tf.multiply(shape, es_tag)), tf.float32)
        mask_fr = tf.cast(tf.equal(source_tags, tf.multiply(shape, fr_tag)), tf.float32)
        
        communication_rewards = res[6]
        fortrans_prob = res[8]

        ## Calculate baselines
        
        # For a single sentence in the batch
        lm_rewards = tf.multiply(mask_es, lm_rewards_es) + tf.multiply(mask_fr, lm_rewards_fr)
        baseline_lm = tf.reduce_mean(lm_rewards)
        lm_rewards = lm_rewards - baseline_lm
        baseline_com = tf.reduce_mean(communication_rewards)
        communication_rewards_b = communication_rewards - baseline_com
        
        # Calculate total reward
        alpha = 0.005
        rewards = tf.add(alpha * lm_rewards, (1 - alpha) * tf.stop_gradient(communication_rewards_b))

        lr1 = 0.0002 # 0.00002
        lr2 = 0.02 # 0.02

        rl_loss = (lr1 * (rewards * fortrans_prob)) + (lr2 * ((1 - alpha) * communication_rewards))
        rl_loss = (-1) * rl_loss


    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      #self.learning_rate = tf.constant(hparams.learning_rate)
      self.learning_rate = tf.constant(8.0)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        tf.summary.scalar("lr", self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)

      loss = 0.0 * self.train_loss +  rl_loss

      gradients = tf.gradients(loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)

      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

      avg_lm_reward = tf.reduce_mean(lm_rewards)
      avg_comm_reward = tf.reduce_mean(communication_rewards)
      avg_for_prob = tf.reduce_mean(fortrans_prob)
      avg_rl_loss = tf.reduce_mean(rl_loss)
      avg_sample_length = tf.reduce_mean(sampling_lengths)

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.train_loss),
          tf.summary.scalar("lm_reward", avg_lm_reward),
          tf.summary.scalar("com_reward", avg_comm_reward),
          tf.summary.scalar("for_prob", avg_for_prob),
          tf.summary.scalar("rl_loss", avg_rl_loss),
          tf.summary.scalar("sample_length", avg_sample_length),
      ] + gradient_norm_summary)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Print trainable variables
    utils.print_out("# Trainable variables")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))


  def get_rewards(self, sampled_logits, sampling_lengths):
      config = ptb_word_lm.LargeConfig()
      config.hidden_size = 512
      lm_vocab_size = self.src_vocab_size - 3
      embedding_matrix = tf.Variable(np.zeros((lm_vocab_size, config.hidden_size), dtype=np.float32),
                                     name="lm_embeddings")
      softmax_weights = tf.Variable(np.zeros((config.hidden_size, lm_vocab_size), dtype=np.float32),
                                    name="softmax_w")
      softmax_bias = tf.Variable(np.zeros((lm_vocab_size), dtype=np.float32), name="softmax_b")

      lm_input_org = sampled_logits 

      lm_input = lm_input_org  
      lm_emb_input = tf.nn.embedding_lookup(embedding_matrix, lm_input)

      sentence_lengths = sampling_lengths
      max_time_steps = tf.reduce_max(sampling_lengths)

      # We decrease the length because we remove the first tag later on
      mask = tf.sequence_mask(tf.subtract(sampling_lengths, 1), max_time_steps-1, dtype=tf.float32)

      cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
      cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(config.num_layers)], state_is_tuple=True)
      state = cell.zero_state(self.batch_size, tf.float32)

      inputs2 = tf.nn.embedding_lookup(embedding_matrix, lm_input)
      with tf.variable_scope("LM_RNN"):
          output, state = tf.nn.dynamic_rnn(cell, inputs2, initial_state=state, sequence_length=sentence_lengths)
      output = tf.reshape(output, [-1, config.hidden_size])
      logits_LM = tf.nn.xw_plus_b(output, softmax_weights, softmax_bias)
      logits_LM = tf.reshape(logits_LM, [self.batch_size, max_time_steps, lm_vocab_size])

      log_probs = tf.nn.log_softmax(logits_LM)
      target_indices = tf.one_hot(lm_input, lm_vocab_size)[:,1:,:]
      log_probs = log_probs[:,:-1,:] * target_indices
      log_probs = tf.reduce_sum(log_probs, axis=2)

      masked_probs = log_probs * mask
      rewards = tf.reduce_sum(masked_probs, axis=1)
      rewards = tf.divide(rewards, tf.cast(sentence_lengths, tf.float32))
      return rewards

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    if (hparams.learning_rate_decay_scheme and
        hparams.learning_rate_decay_scheme == "luong"):
      start_decay_step = int(hparams.num_train_steps / 2)
      decay_steps = int(hparams.num_train_steps / 10)  # decay 5 times
      decay_factor = 0.5
    else:
      start_decay_step = hparams.start_decay_step
      decay_steps = hparams.decay_steps
      decay_factor = hparams.decay_factor
    print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
          "decay_factor %g" %
          (hparams.learning_rate_decay_scheme,
           hparams.start_decay_step, hparams.decay_steps, hparams.decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=hparams.num_units,
            tgt_embed_size=hparams.num_units,
            num_partitions=hparams.num_embeddings_partitions,
            scope=scope,))

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update,
                     self.train_loss,
                     self.predict_count,
                     self.train_summary,
                     self.global_step,
                     self.word_count,
                     self.batch_size])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
                     self.predict_count,
                     self.batch_size])

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss, final_context_state),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: the total loss / batch_size.
        final_context_state: The final state of decoder RNN.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32
    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype) as scope:
      ## Encoder
      encoder_outputs, encoder_state, source_sent = self._build_encoder(hparams)

      ## Decoder
      logits, sample_id, final_context_state, sampling_id, sampling_lengths, max_iterations, fortrans_prob = self._build_decoder(
          encoder_outputs, encoder_state, hparams)

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
          loss = self._compute_loss(logits)
      else:
        loss = None

      ## Backtranslation step
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
          scope.reuse_variables()
          self.backtranslation = True
          self.sampling_ids = sampling_id
          self.sampling_lengths = sampling_lengths + 1

          en_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<en>")), tf.int32)
          es_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<es>")), tf.int32)
          fr_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<fr>")), tf.int32)
          unk_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<unk>")), tf.int32)
          sos_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<s>")), tf.int32)

          shape = tf.ones([self.batch_size], dtype=tf.int32)

          source_tags = source_sent[0]
          mask_es = tf.cast(tf.equal(source_tags, tf.multiply(shape, es_tag)), tf.int32)
          mask_fr = tf.cast(tf.equal(source_tags, tf.multiply(shape, fr_tag)), tf.int32)

          # TODO Target language in the backtranslation is the opposite of the target language in the first step
          tags = mask_es * fr_tag + mask_fr * es_tag

          if self.time_major:
            self.sampling_ids = tf.reshape(self.sampling_ids, [-1])
          else :
            self.sampling_ids = tf.reshape(tf.transpose(self.sampling_ids), [-1])
          self.sampling_ids = tf.concat([tags, self.sampling_ids], 0)
          if self.time_major:
            self.sampling_ids = tf.reshape(self.sampling_ids, [-1, self.batch_size])
          else :
            self.sampling_ids = tf.transpose(tf.reshape(self.sampling_ids, [-1, self.batch_size]))

          # Back-translation
          back_encoder_outputs, back_encoder_state = self._build_encoder(hparams)

          communication_reward = self._build_decoder(
              back_encoder_outputs, back_encoder_state, hparams)

          self.backtranslation = False
      else:
          communication_reward = None
      return logits, loss, final_context_state, sample_id, sampling_id, sampling_lengths, communication_reward, source_sent, fortrans_prob
      #return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                          base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""

    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)

    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    iterator = self.iterator

    ## Decode the source language when backtranslating
    if self.backtranslation:
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state,
                self.sampling_lengths)

            target_input = iterator.source
            
            es_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<es>")), tf.int32)
            fr_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<fr>")), tf.int32)
            sos_tag = tf.cast(self.src_vocab_table.lookup(tf.convert_to_tensor("<s>")), tf.int32)
            
            ones = tf.ones_like(target_input)
            es_matrix = es_tag * ones
            fr_matrix = fr_tag * ones
            subs_es = tf.cast(tf.equal(es_matrix, target_input), tf.int32)
            subs_fr = tf.cast(tf.equal(fr_matrix, target_input), tf.int32)

            es_col = subs_es * es_tag
            target_input = target_input - es_col
            fr_col = subs_fr * fr_tag
            target_input = target_input - fr_col
            sos_col_es = subs_es * sos_tag
            sos_col_fr = subs_fr * sos_tag
            target_input = target_input + sos_col_es + sos_col_fr
            
            if self.time_major:
                target_input = tf.transpose(target_input)
            decoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_decoder, target_input)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, iterator.source_sequence_length,
                time_major=self.time_major)

            # Decoder
            my_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                helper,
                decoder_initial_state, )

            # Dynamic decoding
            outputs, final_context_state, sent_lengths = tf.contrib.seq2seq.dynamic_decode(
                my_decoder,
                output_time_major=self.time_major,
                swap_memory=True,
                scope=decoder_scope)

            sample_id = outputs.sample_id

            # Note: there's a subtle difference here between train and inference.
            # We could have set output_layer when create my_decoder
            #   and shared more code between train and inference.
            # We chose to apply the output_layer to all timesteps for speed:
            #   10% improvements for small models & 20% for larger ones.
            # If memory is a concern, we should apply output_layer per timestep.
            device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
            with tf.device(model_helper.get_device_str(device_id, num_gpus)):
                logits = self.output_layer(outputs.rnn_output)

            log_probs = tf.nn.log_softmax(logits)

            target_indices = tf.one_hot(target_input[1:,:], self.src_vocab_size)
            log_probs = log_probs[:-1,:,:] * target_indices
            log_probs = tf.reduce_sum(log_probs, axis=2)
            log_probs = tf.transpose(log_probs)

            max_length = tf.reduce_max(sent_lengths)
            mask = tf.sequence_mask(sent_lengths, max_length-1, dtype=tf.float32)
            masked_probs = log_probs * mask
            rewards = tf.reduce_sum(masked_probs, axis=1)
            rewards = tf.divide(rewards, tf.cast(sent_lengths, tf.float32))
        return rewards


    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, iterator.source_sequence_length)

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input

        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
          logits = self.output_layer(outputs.rnn_output)

      ## Inference
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)

        else:
          # Helper
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

      ## Get sampling translation
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
          with tf.variable_scope("sampling") as sampling_scope:
              start_tokens = tf.fill([self.batch_size], tgt_sos_id)
              end_token = tgt_eos_id

              # Sampling helper
              sampling_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding_decoder, start_tokens, end_token, softmax_temperature=0.002)
              sampling_decoder = tf.contrib.seq2seq.BasicDecoder(
                  cell,
                  sampling_helper,
                  decoder_initial_state,
                  output_layer=self.output_layer
              )

              # Sampling dynamic decoding
              sampling_outputs, sampling_final_context_state, sampling_seq_lengths = tf.contrib.seq2seq.dynamic_decode(
                  sampling_decoder,
                  maximum_iterations=maximum_iterations,
                  output_time_major=self.time_major,
                  swap_memory=True,
                  scope=sampling_scope)
              sampling_sample_id = sampling_outputs.sample_id

              sampling_sample_id = tf.stop_gradient(sampling_sample_id)

              sample_logits = sampling_outputs.rnn_output
            
              log_probs = tf.nn.log_softmax(sample_logits)
              sample_indices = tf.one_hot(sampling_sample_id, self.src_vocab_size) 
              log_probs = log_probs * sample_indices
              log_probs = tf.reduce_sum(log_probs, axis=2)
              log_probs = tf.transpose(log_probs)

              max_length = tf.reduce_max(sampling_seq_lengths)
              mask = tf.sequence_mask(sampling_seq_lengths, max_length, dtype=tf.float32)
              masked_probs = log_probs * mask
              fortrans_prob = tf.reduce_sum(masked_probs, axis=1)
              fortrans_prob = tf.divide(fortrans_prob, tf.cast(sampling_seq_lengths, tf.float32))
      else :
          sampling_sample_id = None
          sampling_seq_lengths = None
          fortrans_prob = None

    # return logits, sample_id, final_context_state
    return logits, sample_id, final_context_state, sampling_sample_id, sampling_seq_lengths, maximum_iterations, fortrans_prob

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder
        and the intial state of the decoder RNN.
    """
    pass

  def _compute_loss(self, logits):
    """Compute optimization loss."""
    target_output = self.iterator.target_output
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(self.batch_size)
    return loss

  def _get_infer_summary(self, hparams):
    return tf.no_op()

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    return sess.run([
        self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
    ])

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    _, infer_summary, _, sample_words = self.infer(sess)

    # make sure outputs is of shape [batch_size, time]
    if self.time_major:
      sample_words = sample_words.transpose()
    return sample_words, infer_summary


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder(self, hparams):
    """Build an encoder."""
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    iterator = self.iterator

    source = iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(
          self.embedding_encoder, source)

      # Encoder_outpus: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(
            hparams, num_layers, num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=self.time_major,
            swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=iterator.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
    return encoder_outputs, encoder_state

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state

class PTBInput(object):
  """The input data."""
  def ptb_producer(self, raw_data, batch_size, num_steps, name=None):

    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
      #raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

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


  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((tf.size(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = self.ptb_producer(
        data, batch_size, num_steps, name=name)


