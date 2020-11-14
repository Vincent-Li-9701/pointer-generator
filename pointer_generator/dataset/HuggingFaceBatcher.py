from datasets import load_dataset
from pointer_generator.utils.dataset import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN, SENTENCE_END, SENTENCE_STA
from transformers import BasicTokenizer
from pointer_generator.utils.dataset import ExampleHF, Batch
from pointer_generator.utils.utils import abstract2sents
from threading import Thread
import queue
import tensorflow as tf
import random
import time


class HuggingFaceBatcher():
    BATCH_QUEUE_MAX = 50  # max number of batches the batch_queue can hold

    def __init__(self, dataset, vocab, tokenizer, batch_size, single_pass=False, mode='train', shuffle=True):
        self.dataset = dataset
        self._vocab = vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.single_pass = single_pass
        self.mode = mode
        self.shuffle = shuffle

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # num threads to fill example queue
            self._num_batch_q_threads = 1  # num threads to fill batch queue
            self._bucketing_cache_size = 1  # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.compat.v1.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self.single_pass and self._finished_reading:
                tf.compat.v1.logging.info(
                    "Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        input_gen = self.pair_generator()

        while True:
            try:
                (article,
                 abstract) = input_gen.__next__()  # read the next example from file. article and abstract are both strings.
            except StopIteration:  # if there are no more examples:
                tf.compat.v1.logging.info(
                    "The example generator for this example queue filling thread has exhausted data.")
                if self.single_pass:
                    tf.compat.v1.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        "single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in abstract2sents(
                abstract, encode=False)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
            example = ExampleHF(article, abstract_sentences, self._vocab)
            self._example_queue.put(example)

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len,
                                reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self.single_pass:
                    random.shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.compat.v1.logging.info(
                'Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.compat.v1.logging.error(
                        'Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.compat.v1.logging.error(
                        'Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def index_generator(self):
        counter = len(self.dataset)
        idx_list = list(range(len(self.dataset)))
        while True:
            if counter >= len(self.dataset):
                counter = 0
                if self.shuffle:
                    random.shuffle(idx_list)
            yield idx_list[counter]
            counter += 1

    def pair_generator(self):
        index_generator = self.index_generator()
        while True:
            idx = next(index_generator)
            article, summary = self.dataset[idx]
            article_text = " ".join(self.tokenizer.encode(article).tokens)
            abstract_text = " ".join(
                ["%s %s %s" % (SENTENCE_STA, " ".join(self.tokenizer.encode(sent).tokens), SENTENCE_END)
                 for sent in summary.split("\n")])
            if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                # tf.logging.warning('Found an example with empty article text. Skipping it.')
                continue
            else:
                yield article_text, abstract_text
