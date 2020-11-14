from pointer_generator.dataset.HuggingFaceDataset import name_to_HFDS
from pointer_generator.dataset.HuggingFaceBatcher import HuggingFaceBatcher
import random


class MetaBatcher(object):
    """
    Simple meta loader -- every dataset is a new task
    """
    def __init__(self, num_samples_per_task, vocab, tokenizer=None, datasets="all", split="train", **batcher_args):
        if datasets == "all":
            datasets = list(name_to_HFDS.keys())
        self.ds_names = datasets

        assert vocab is not None, "No vocab file provided"
        self._vocab = vocab
        self.tokenizer = tokenizer

        self.K = num_samples_per_task
        self.batcher = []
        assert split in ["train", "test", "validation"]
        for name in datasets:
            self.batcher.append(
                HuggingFaceBatcher(dataset=name_to_HFDS[name](split=split),
                    vocab=self._vocab, tokenizer=tokenizer, batch_size=self.K, **batcher_args))

    def next_batch(self):
        random_task = random.randint(0, len(self.ds_names)-1)
        return self.batcher[random_task].next_batch()
