from typing import List
from pointer_generator.utils.dataset import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, SENTENCE_END, SENTENCE_STA
import csv
from transformers import BasicTokenizer
from pointer_generator.dataset.HuggingFaceDataset import name_to_HFDS
from pointer_generator.utils.config import vocab_cache_dir
import os
from collections import Counter


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


def combine_vocab_files(input_vocab_files, output_vocab_file, max_size):
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
                vocab_counter[w] += freq

    print("Writing vocab file...")
    file_path = os.path.join(vocab_cache_dir, output_vocab_file)
    with open(file_path, 'w') as writer:
        for word, count in vocab_counter.most_common(max_size):
            writer.write(word + ' ' + str(count) + '\n')
    print(f"Finished writing vocab file {file_path}. "
          f"The least common word is {word} with frequency {count}")


if __name__ == '__main__':
    # build_vocab_from_HFDS("cnn_dailymail", 100000, vocab_name="cnn_dailymail.txt")
    build_vocab_from_HFDS("xsum", 100000, vocab_name="xsum.txt")
