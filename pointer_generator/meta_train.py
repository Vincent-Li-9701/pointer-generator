# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import tensorflow as tf

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import higher

from pointer_generator.models.model import Model
from pointer_generator.utils import config
from pointer_generator.utils.dataset import Vocab, WordPieceVocab
from pointer_generator.utils.utils import get_input_from_batch
from pointer_generator.utils.utils import get_output_from_batch
from pointer_generator.utils.utils import calc_running_avg_loss
from pointer_generator.dataset.HuggingFaceDataset import CNNDailyMailDataset, XSumDataset
from pointer_generator.dataset.HuggingFaceBatcher import HuggingFaceBatcher
from pointer_generator.dataset.MetaBatcher import MetaBatcher
from tokenizers import Tokenizer

tf.config.set_visible_devices([], 'GPU')
use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        # self.vocab = Vocab(config.vocab_path, config.vocab_size)
        # self.batcher = Batcher(self.vocab, config.train_data_path,
        #                        config.batch_size, single_pass=False, mode='train')
        '''Train on one dataset'''
        # dataset = CNNDailyMailDataset(split="train")
        # self.batcher = HuggingFaceBatcher(dataset=dataset, vocab=self.vocab, batch_size=2)
        '''Meta Training'''
        if config.use_wordpiece_vocab:
            self.vocab = WordPieceVocab(os.path.join(config.vocab_cache_dir, config.meta_vocab_file))
        else:
            self.vocab = Vocab(os.path.join(config.vocab_cache_dir, config.meta_vocab_file), max_size=config.meta_vocab_size)
        tokenizer = Tokenizer.from_file(os.path.join(config.vocab_cache_dir, config.meta_tokenizer_file))
        self.batcher = MetaBatcher(num_samples_per_task=config.meta_train_batch_size * (config.num_train_batches + config.num_test_batches),
                                   datasets=config.meta_train_datasets,
                                   vocab=self.vocab,
                                   tokenizer=tokenizer,
                                   split="train")

        time.sleep(10)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.create_file_writer(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'model_dict': self.model.state_dict(),
            'optimizer': self.meta_optimizer.state_dict(),
            'inner_optimizer': self.inner_optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_path=None):
        self.model = Model(model_path, is_tran=config.tran)
        initial_lr = config.lr_coverage if config.is_coverage else config.lr

        params = list(self.model.parameters())
        total_params = sum([param[0].nelement() for param in params])
        print('The Number of params of model: %.3f million' % (total_params / 1e6))  # million

        # self.meta_optimizer = optim.Adagrad(self.model.parameters(), lr=0.35, weight_decay=0.1,
        #                                     initial_accumulator_value=config.adagrad_init_acc)
        # self.inner_optimizer = optim.Adagrad(self.model.parameters(), lr=0.45, weight_decay=0.1,
        #                                      initial_accumulator_value=config.adagrad_init_acc)

        self.params = list(self.model.parameters())
        self.meta_optimizer = optim.Adam(self.params, lr=1e-4, weight_decay=1e-5)
        self.inner_optimizer = optim.Adam(self.params, lr=1e-4, weight_decay=0)

        start_iter, start_loss = 0, 0

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            #start_iter = state['iter']
            #start_loss = state['current_loss']

            if not config.is_coverage:

                """
                if 'model_dict' in state.keys():
                    self.meta_optimizer.load_state_dict(state['optimizer'])
                    if 'inner_optimizer' in state.keys():
                        self.inner_optimizer.load_state_dict(state['inner_optimizer'])
                else:
                    pass
                    # self.meta_optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.meta_optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                    for state in self.inner_optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                """
        return start_iter, start_loss

    def meta_train_one_batch(self, batch):
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda)

        def split_data(data):
            if data is None:
                return [None] * (config.num_train_batches + config.num_test_batches)
            else:
                #print(data.shape, type(data))
                d = torch.split(data, split_size_or_sections = 2)# (config.num_train_batches + config.num_test_batches))
                #print(d[0].shape)
                return d

        enc_batch = split_data(enc_batch)
        enc_pos = split_data(enc_pos)
        enc_padding_mask = split_data(enc_padding_mask)
        enc_batch_extend_vocab= split_data(enc_batch_extend_vocab)
        extra_zeros = split_data(extra_zeros)
        coverage = split_data(coverage)
        c_t = split_data(c_t)
        dec_batch = split_data(dec_batch)
        dec_lens = split_data(dec_lens)
        dec_pos = split_data(dec_pos)
        dec_padding_mask = split_data(dec_padding_mask)
        tgt_batch = split_data(tgt_batch)

        data_pointer = config.num_train_batches
        # test before meta rain
        with torch.no_grad():
            meta_test_loss = 0
            for i in range(config.num_test_batches):
                enc_out, enc_fea, enc_h = self.model.encoder(enc_batch[data_pointer], enc_pos[data_pointer])
                s_t = self.model.reduce_state(enc_h)
                step_losses, cove_losses = [], []
                for di in range(min(max_dec_len, config.max_dec_steps)):
                    y_t = dec_batch[data_pointer][:, di]  # Teacher forcing
                    c = c_t[data_pointer]
                    final_dist, s_t, c, attn_dist, p_gen, next_coverage = \
                        self.model.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask[data_pointer], c,
                                     extra_zeros[data_pointer], enc_batch_extend_vocab[data_pointer],
                                     coverage[data_pointer], di)
                    tgt = tgt_batch[data_pointer][:, di]
                    step_mask = dec_padding_mask[data_pointer][:, di]
                    gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()
                    step_loss = -torch.log(gold_probs + config.eps)
                    if config.is_coverage:
                        step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                        step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                        cove_losses.append(step_coverage_loss * step_mask)
                        coverage = next_coverage

                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)

                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                batch_avg_loss = sum_losses / dec_lens[data_pointer]
                l = torch.mean(batch_avg_loss)
                meta_test_loss += l
                data_pointer += 1

        prev_loss = meta_test_loss / config.num_test_batches

        data_pointer = 0
        inner_loss = []
        with higher.innerloop_ctx(self.model, self.inner_optimizer,
                                  copy_initial_weights=False) as (fnet, diffopt):

            for i in range(config.num_train_batches):
                # the method will always be used for trianing
                enc_out, enc_fea, enc_h = fnet.encoder(enc_batch[data_pointer], enc_pos[data_pointer])
                s_t = fnet.reduce_state(enc_h)
                step_losses, cove_losses = [], []
                for di in range(min(max_dec_len, config.max_dec_steps)):
                    y_t = dec_batch[data_pointer][:, di]  # Teacher forcing

                    c = c_t[data_pointer]
                    # TODO: extra_zeros and enc_batch_extend_vocab can be None if pg is enabled. need to handle that
                    # coverage is zero when converage is enabled
                    final_dist, s_t, c, attn_dist, p_gen, next_coverage = \
                        fnet.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask[data_pointer], c,
                                     extra_zeros[data_pointer], enc_batch_extend_vocab[data_pointer], coverage[data_pointer], di)

                    tgt = tgt_batch[data_pointer][:, di]
                    step_mask = dec_padding_mask[data_pointer][:, di]
                    gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()
                    step_loss = -torch.log(gold_probs + config.eps)
                    if config.is_coverage:
                        step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                        step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                        cove_losses.append(step_coverage_loss * step_mask)
                        coverage = next_coverage

                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)

                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                batch_avg_loss = sum_losses / dec_lens[data_pointer]
                meta_train_loss = torch.mean(batch_avg_loss)
                diffopt.step(meta_train_loss)
                inner_loss.append(meta_train_loss.item())
                #print("iteration: {} meta_train_loss: {}".format(i, meta_train_loss))
                data_pointer += 1
            inner_loss.append(0)

            ## meta test
            meta_test_loss = 0
            for i in range(config.num_test_batches):
                enc_out, enc_fea, enc_h = fnet.encoder(enc_batch[data_pointer], enc_pos[data_pointer])
                s_t = fnet.reduce_state(enc_h)
                step_losses, cove_losses = [], []
                for di in range(min(max_dec_len, config.max_dec_steps)):
                    y_t = dec_batch[data_pointer][:, di]  # Teacher forcing
                    c = c_t[data_pointer]
                    final_dist, s_t, c, attn_dist, p_gen, next_coverage = \
                        fnet.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask[data_pointer], c,
                                    extra_zeros[data_pointer], enc_batch_extend_vocab[data_pointer], coverage[data_pointer], di)
                    tgt = tgt_batch[data_pointer][:, di]
                    step_mask = dec_padding_mask[data_pointer][:, di]
                    gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()
                    step_loss = -torch.log(gold_probs + config.eps)
                    if config.is_coverage:
                        step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                        step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                        cove_losses.append(step_coverage_loss * step_mask)
                        coverage = next_coverage

                    step_loss = step_loss * step_mask
                    step_losses.append(step_loss)

                sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
                batch_avg_loss = sum_losses / dec_lens[data_pointer]
                l = torch.mean(batch_avg_loss)
                meta_test_loss += l

                #print("iteration: {} meta_test_loss: {}".format(i, l))
                inner_loss.append(l.item())
                data_pointer += 1

        self.meta_optimizer.zero_grad()
        meta_test_loss = meta_test_loss/config.num_test_batches
        meta_test_loss.backward()
        self.meta_optimizer.step()

        """
        if config.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = cove_losses / dec_lens
            batch_cove_loss = torch.mean(batch_cove_loss)
            return meta_test_loss.item(), batch_cove_loss.item()
        """

        return prev_loss.item(), meta_train_loss.item(), meta_test_loss.item(), inner_loss

    def run(self, n_iters, model_path=None):
        iter, running_avg_loss = self.setup_train(model_path)
        start = time.time()
        interval = 50

        prev_loss = 0
        train_loss = 0
        test_loss = 0

        self.save_model(running_avg_loss, iter)
        while iter < n_iters:
            batch = self.batcher.next_batch()
            pl, trainl, testl, inner_loss = self.meta_train_one_batch(batch)
            prev_loss += pl
            train_loss += trainl
            test_loss += testl
            running_avg_loss = calc_running_avg_loss(test_loss, running_avg_loss, self.summary_writer, iter)

            if iter % interval == 0:
                self.summary_writer.flush()
                print(
                    'step: %d, second: %.2f , meta train loss: %f, meta test prev loss: %f, post loss: %f, diff:%f' % \
                    (iter, time.time() - start, train_loss/interval, prev_loss/interval, test_loss/interval, (prev_loss-test_loss)/interval))
                prev_loss = 0
                train_loss = 0
                test_loss = 0
                start = time.time()
                #print(inner_loss)
            if iter % 500 == 0:
                self.save_model(running_avg_loss, iter)

            iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.run(config.max_iterations, args.model_path)
