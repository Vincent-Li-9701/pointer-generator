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
        self.batcher = MetaBatcher(num_samples_per_task=config.meta_train_K,
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

        self.meta_optimizer = optim.Adagrad(self.model.parameters(), lr=initial_lr, weight_decay=0.1,
                                            initial_accumulator_value=config.adagrad_init_acc)
        self.inner_optimizer = optim.Adagrad(self.model.parameters(), lr=initial_lr, weight_decay=0.1,
                                             initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                if 'model_dict' in state.keys():
                    self.meta_optimizer.load_state_dict(state['optimizer'])
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

        return start_iter, start_loss

    def meta_train_one_batch(self, batch):
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda)

        def split_data(data):
            if data is None:
                return None, None
            else:
                return data[:-1], data[-1:]

        mtr_extra_zeros, mte_extra_zeros = split_data(extra_zeros)
        mtr_enc_batch_extend_vocab, mte_enc_batch_extend_vocab = split_data(enc_batch_extend_vocab)
        mtr_coverage, mte_coverage = split_data(coverage)
        mtr_c_t, mte_c_t = split_data(c_t)

        with higher.innerloop_ctx(self.model, self.inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
            for _ in range(config.num_inner_loops):
                # the method will always be used for trianing
                enc_out, enc_fea, enc_h = fnet.encoder(enc_batch[:-1], enc_pos[:-1])
                s_t = fnet.reduce_state(enc_h)
                step_losses, cove_losses = [], []
                c_t = mtr_c_t
                for di in range(min(max_dec_len, config.max_dec_steps)):
                    y_t = dec_batch[:-1, di]  # Teacher forcing

                    # TODO: extra_zeros and enc_batch_extend_vocab can be None if pg is enabled. need to handle that
                    # coverage is zero when converage is enabled
                    final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = \
                        fnet.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask[:-1], c_t,
                                     mtr_extra_zeros, mtr_enc_batch_extend_vocab, mtr_coverage, di)
                    tgt = tgt_batch[:-1, di]
                    step_mask = dec_padding_mask[:-1, di]
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
                batch_avg_loss = sum_losses / dec_lens[:-1]
                meta_train_loss = torch.mean(batch_avg_loss)
                diffopt.step(meta_train_loss)

            ## meta test
            enc_out, enc_fea, enc_h = fnet.encoder(enc_batch[-1:], enc_pos[-1:])
            s_t = fnet.reduce_state(enc_h)
            c_t = mte_c_t
            step_losses, cove_losses = [], []
            for di in range(min(max_dec_len, config.max_dec_steps)):
                y_t = dec_batch[-1:, di]  # Teacher forcing
                final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = \
                    fnet.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask[-1:], c_t,
                                 mte_extra_zeros, mte_enc_batch_extend_vocab, mte_coverage, di)
                tgt = tgt_batch[-1:, di]
                step_mask = dec_padding_mask[-1:, di]
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
            batch_avg_loss = sum_losses / dec_lens[-1:]
            meta_test_loss = torch.mean(batch_avg_loss)

        self.meta_optimizer.zero_grad()
        meta_test_loss.backward()
        self.meta_optimizer.step()

        if config.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = cove_losses / dec_lens
            batch_cove_loss = torch.mean(batch_cove_loss)
            return loss.item(), batch_cove_loss.item()

        return meta_train_loss.item(), meta_test_loss.item()

    def run(self, n_iters, model_path=None):
        iter, running_avg_loss = self.setup_train(model_path)
        start = time.time()
        interval = 100

        self.save_model(running_avg_loss, iter)
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss, cove_loss = self.meta_train_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % interval == 0:
                self.summary_writer.flush()
                print(
                    'step: %d, second: %.2f , meta train loss: %f, meta test loss: %f' % (iter, time.time() - start, loss, cove_loss))
                start = time.time()
            if iter % 5000 == 0:
                self.save_model(running_avg_loss, iter)


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
