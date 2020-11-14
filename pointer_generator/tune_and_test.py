# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import torch
import argparse
import tensorflow as tf
from torch.autograd import Variable

from models.model import Model
from utils import utils
from utils.dataset import Batcher
from utils.dataset import Vocab
from utils import dataset, config
from pointer_generator.utils.utils import get_input_from_batch
from pointer_generator.utils.utils import get_output_from_batch
from pointer_generator.utils.utils import calc_running_avg_loss
import torch.optim as optim
from utils.utils import write_for_rouge, rouge_eval, rouge_log
from pointer_generator.dataset.MetaBatcher import MetaBatcher
from pointer_generator.utils.dataset import Vocab, WordPieceVocab
from tokenizers import Tokenizer

tf.config.set_visible_devices([], 'GPU')
use_cuda = config.use_gpu and torch.cuda.is_available()


class Train(object):
    def __init__(self, dataset):
        if config.use_wordpiece_vocab:
            self.vocab = WordPieceVocab(os.path.join(config.vocab_cache_dir, config.meta_vocab_file))
        else:
            self.vocab = Vocab(os.path.join(config.vocab_cache_dir, config.meta_vocab_file), max_size=config.meta_vocab_size)
        tokenizer = Tokenizer.from_file(os.path.join(config.vocab_cache_dir, config.meta_tokenizer_file))
        self.batcher = MetaBatcher(num_samples_per_task=config.meta_train_K,
                                   datasets=config.meta_test_datasets,
                                   vocab=self.vocab,
                                   tokenizer=tokenizer,
                                   split="train")

        """
        # seems batch splits are different...
        # and for weird reasons give you terrible results if you train cnn with metabatcher
        # could metabatcher be wrong??? (please don't!
        self.batcher = Batcher(self.vocab, config.train_data_path,
                               config.batch_size, single_pass=False, mode='train')
        """
        time.sleep(10)
        train_dir = os.path.join(config.tmp_dir, 'test_%d' % (int(time.time())))
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
            'optimizer': self.optimizer.state_dict(),
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

        self.optimizer = optim.Adagrad(self.model.parameters(), lr=initial_lr,
                                             initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            if not config.is_coverage:
                if 'model_dict' in state.keys():
                    self.optimizer.load_state_dict(state['inner_optimizer'])
                else:
                    pass
                    # self.meta_optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        if not config.tran:
            enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_lens)
        else:
            enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_pos)

        s_t = self.model.reduce_state(enc_h)

        step_losses, cove_losses = [], []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t, c_t, attn_dist, p_gen, next_coverage = \
                self.model.decoder(y_t, s_t, enc_out, enc_fea, enc_padding_mask, c_t,
                                   extra_zeros, enc_batch_extend_vocab, coverage, di)
            tgt = tgt_batch[:, di]
            step_mask = dec_padding_mask[:, di]
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
        batch_avg_loss = sum_losses / dec_lens
        loss = torch.mean(batch_avg_loss)

        loss.backward()
        self.optimizer.step()

        if config.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = cove_losses / dec_lens
            batch_cove_loss = torch.mean(batch_cove_loss)
            return loss.item(), batch_cove_loss.item()

        return loss.item(), 0.

    def run(self, n_iters, model_path=None):
        iter, running_avg_loss = self.setup_train(model_path)
        start = time.time()
        interval = max(100, n_iters//10)

        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss, cove_loss = self.train_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % interval == 0:
                self.summary_writer.flush()
                print(
                    'step: %d, second: %.2f , loss: %f, cover_loss: %f' % (iter, time.time() - start, loss, cove_loss))
                start = time.time()

        self.save_model(running_avg_loss, iter)

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.state = state
        self.context = context
        self.coverage = coverage
        self.log_probs = log_probs

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

class Test(object):
    def __init__(self, model_file_path, model, vocab, dataset):

        model_name = os.path.basename(model_file_path)
        self._test_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._test_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._test_dir, 'rouge_dec')
        for p in [self._test_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        if config.use_wordpiece_vocab:
            self.vocab = WordPieceVocab(os.path.join(config.vocab_cache_dir, config.meta_vocab_file))
        else:
            self.vocab = Vocab(os.path.join(config.vocab_cache_dir, config.meta_vocab_file), max_size=config.meta_vocab_size)
        tokenizer = Tokenizer.from_file(os.path.join(config.vocab_cache_dir, config.meta_tokenizer_file))
        self.batcher = MetaBatcher(num_samples_per_task=config.meta_test_K,
                                   datasets=[dataset],
                                   vocab=self.vocab,
                                   tokenizer=tokenizer,
                                   split="test",
                                   mode="decode")

        time.sleep(15)

        self.model = model

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        # single example repeated across the batch
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, c_t, coverage = \
            get_input_from_batch(batch, use_cuda)
        
        print(enc_batch.shape)

        enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_pos)
        s_t = self.model.reduce_state(enc_h)

        dec_h, dec_c = s_t  # b x hidden_dim
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(config.BOS_TOKEN)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t[0],
                      coverage=(coverage[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        steps = 0
        results = []
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(config.UNK_TOKEN) \
                             for t in latest_tokens]
            y_t = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t = y_t.cuda()
            all_state_h = [h.state[0] for h in beams]
            all_state_c = [h.state[1] for h in beams]
            all_context = [h.context for h in beams]

            s_t = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t = torch.stack(all_context, 0)

            coverage_t = None
            if config.is_coverage:
                all_coverage = [h.coverage for h in beams]
                coverage_t = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t, s_t,
                                                                                    enc_out, enc_fea,
                                                                                    enc_padding_mask, c_t,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(config.EOS_TOKEN):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def run(self, num_to_eval, print_result):

        counter = 0
        start = time.time()
        batch = self.batcher.next_batch()
        interval = max(100, num_to_eval // 10)
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            """
            print(output_ids)
            input()
            print(batch.art_oovs[0])
            input()
            """
            decoded_words = utils.outputids2words(output_ids, self.vocab,
                                                  (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(dataset.EOS_TOKEN)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            if print_result:
                print("pred:", " ".join(decoded_words))
                print("gt:", " ".join(original_abstract_sents))
                print("=====")

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % interval == 0:
                print('%d example in %d sec' % (counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()
            if num_to_eval > 0 and counter > num_to_eval:
                break

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._test_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("-m",
                        dest="model_path",
                        required=True,
                        default=None,
                        help="Model file for retraining (default: None).")
    parser.add_argument("-n",
                        dest="num_to_eval",
                        required=False,
                        default=0,
                        help="Evaluate the first n examples")
    parser.add_argument("-p",
                        dest="print",
                        required=False,
                        default=False,
                        help="whether to print")
    parser.add_argument("-d",
                        dest="dataset",
                        required=False,
                        default=config.meta_test_datasets,
                        help="test dataset (folder) name")
    parser.add_argument("-i",
                        dest="max_iterations",
                        required=False,
                        default=1,
                        help="number of iterations to pretrain")
    args = parser.parse_args()

    # load model
    train_processor = Train(args.dataset)
    # train
    train_processor.run(int(args.max_iterations), args.model_path)
    # clean up train
    # todo: is this necessary?
    # test
    test_processor = Test(args.model_path, train_processor.model, train_processor.vocab, args.dataset)
    test_processor.run(int(args.num_to_eval), bool(args.print))

