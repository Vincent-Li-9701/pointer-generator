from typing import List
from pointer_generator.utils.dataset import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, SENTENCE_END, SENTENCE_STA
import csv
from transformers import BasicTokenizer
from pointer_generator.dataset.HuggingFaceDataset import name_to_HFDS
from pointer_generator.utils.config import vocab_cache_dir, UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
import os
from collections import Counter
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer



class BaseVocab(object):
    def __init__(self, max_size):
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0     # keeps track of total number of words in the Vocab
        self.max_size = max_size

        for w in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
            self.count += 1

    def build_from_tokens(self, tokens: List[str]):
        for w in tokens:
            if w not in self.word2idx:
                if self.count >= self.max_size:
                    # print("WARNING: maximum size of vocab is reached")
                    break
                self.word2idx[w] = self.count
                self.idx2word[self.count] = w
                self.count += 1
        # print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
        #         self.count, self.idx2word[self.count - 1]))

    def build_from_file(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                if self.count >= self.max_size:
                    print("WARNING: maximum size of vocab is reached")
                    break
                w = line[:-1]
                self.word2idx[w] = self.count
                self.idx2word[self.count] = w
                self.count += 1
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
          self.count, self.idx2word[self.count - 1]))

    def write_to_file(self, filename):
        with open(filename, "w") as f:
            for i in range(self.count):
                f.write(f"{self.idx2word[i]}\n")
        print(f"Wrote {self.count} words into file {filename}")

    def word2id(self, word):
        if word not in self.word2idx:
            return self.word2idx[UNK_TOKEN]
        return self.word2idx[word]

    def id2word(self, word_id):
        if word_id not in self.idx2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.idx2word[word_id]

    def size(self):
        return self.count

    def write_metadata(self, path):
        print( "Writing word embedding metadata file to %s..." % (path))
        with open(path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                writer.writerow({"word": self.idx2word[i]})


def build_vocab_from_HFDS(dataset_names, max_size, vocab_name, split="train"):
    split = ["train", "test", "validation"] if split == "all" else [split]
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    tokenizer = BasicTokenizer()
    vocab_counter = Counter()

    for dataset in dataset_names:
        for s in split:
            ds = name_to_HFDS[dataset](split=s)
            for i in range(len(ds)):
                print(f"\r {dataset} {s} -- {i} / {len(ds)}", end="", flush=True)
                article, summary = ds[i]
                art_tokens = tokenizer.tokenize(article)
                sum_tokens = tokenizer.tokenize(summary)
                sum_tokens = [t for t in sum_tokens if
                              t not in [SENTENCE_STA,
                                        SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + sum_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)
            print()

    print(f"Finished reading datasets.")

    print("Writing vocab file...")
    file_path = os.path.join(vocab_cache_dir, vocab_name)
    with open(file_path, 'w') as writer:
        for word, count in vocab_counter.most_common(max_size):
            writer.write(word + ' ' + str(count) + '\n')
    print(f"Finished writing vocab file {file_path}. "
          f"The least common word is {word} with frequency {count}")


def combine_vocab_files(input_vocab_files, output_vocab_file, max_size, criteria="task"):
    assert criteria in ["task", "frequency"]
    vocab_counter = Counter()
    for vocab_file in input_vocab_files:
        print(vocab_file)
        vocab_path = os.path.join(vocab_cache_dir, vocab_file)
        with open(vocab_path, 'r') as fin:
            for line in fin:
                items = line.split()
                if len(items) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s' % line.strip())
                    continue
                w, freq = items
                if w in [SENTENCE_STA, SENTENCE_END, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [BOS] and [EOS] shouldn\'t be in the vocab file, but %s is' % w)
                if criteria == "task":
                    vocab_counter[w] += 1
                else:
                    vocab_counter[w] += int(freq)

    print("Writing vocab file...")
    file_path = os.path.join(vocab_cache_dir, output_vocab_file)
    with open(file_path, 'w') as writer:
        for word, count in vocab_counter.most_common(max_size):
            writer.write(word + ' ' + str(count) + '\n')
    print(f"Finished writing vocab file {file_path}. "
          f"The least common word is {word} with frequency {count}")


def build_wordpiece_vocab_training_files(datasets, filename, split="train"):
    if datasets == "all":
        datasets = list(name_to_HFDS.keys())
    if isinstance(datasets, str):
        datasets = [datasets]
    split = ["train", "test", "validation"] if split == "all" else [split]

    with open(os.path.join(vocab_cache_dir, filename), "w") as fout:
        for dataset in datasets:
            for s in split:
                ds = name_to_HFDS[dataset](split=s)
                for i in range(len(ds)):
                    print(f"\r {dataset} {s} -- {i} / {len(ds)}", end="", flush=True)
                    article, summary = ds[i]
                    fout.write(article)
                    fout.write("\n")
                    fout.write(summary)
                    fout.write("\n")


def train_wordpiece_tokenizer(train_files, tokenizer_filename, vocab_size=50000):
    if isinstance(train_files, str):
        train_files = [train_files]
    bert_tokenizer = Tokenizer(WordPiece(unk_token=UNK_TOKEN))
    bert_tokenizer.normalizer = normalizers.Sequence(
        [NFD(), Lowercase(), StripAccents()])
    bert_tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=[UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]
    )
    bert_tokenizer.train(trainer, train_files)

    model_files = bert_tokenizer.model.save(vocab_cache_dir, tokenizer_filename)
    bert_tokenizer.model = WordPiece.from_file(*model_files, unk_token=UNK_TOKEN)

    bert_tokenizer.save(os.path.join(vocab_cache_dir, tokenizer_filename))


if __name__ == '__main__':
    # build_vocab_from_HFDS("cnn_dailymail", 100000, vocab_name="cnn_dailymail.txt")
    # combine_vocab_files(["reddit_tifu_short.txt", "reddit_tifu.txt"], "reddit_tifu_all.txt", max_size=100000, criteria="frequency")
    # combine_vocab_files(["aeslc.txt", "billsum.txt", "cnn_dailymail.txt", "gigaword.txt", "multi_news.txt", "reddit_tifu_all.txt", "xsum.txt"],
    #                     "vocab_7ds_5w.txt", max_size=50000, criteria="task")
    # combine_vocab_files(
    #     ["billsum.txt", "cnn_dailymail.txt", "gigaword.txt",
    #      "multi_news.txt", "xsum.txt"],
    #     "vocab_5ds_5w.txt", max_size=50000, criteria="task")


    # TEST BERT WORDPIECE
    # build_wordpiece_vocab_training_files("all", "training/5ds.txt")
    # files = ["/scr-ssd/yanjunc/course/cs330/dailymail_raw/stories/7777022b710224353e4e12f6466a1b187d1996da.story",
    #          "/scr-ssd/yanjunc/course/cs330/huggingface/vocab/training/specials"]
    # train_wordpiece_tokenizer(os.path.join(vocab_cache_dir, "training/5ds.txt"), "wp_5ds_5w", vocab_size=50000)
    bert_tokenizer = Tokenizer.from_file(os.path.join(vocab_cache_dir, "wp_5ds_5w"))
    output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.\n\n\n\n")
    print(output.tokens)
    # with open(files[0], "r") as fin:
    #     data = " ".join(fin.readlines())
    #     encoding = tokenizer.encode(data)
    #     print(encoding.tokens)
