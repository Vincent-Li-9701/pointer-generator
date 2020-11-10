from datasets import load_dataset
from pointer_generator.utils.config import dataset_cache_dir


class CNNDailyMailDataset(object):
    def __init__(self, split="train"):
        assert split in ["train", "test", "validation"]
        self.dataset = load_dataset("cnn_dailymail",  '3.0.0', cache_dir=dataset_cache_dir)[split]

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        pair = self.dataset[index]
        return pair["article"], pair["highlights"]


class XSumDataset(object):
    def __init__(self, split="train"):
        assert split in ["train", "test", "validation"]
        self.dataset = load_dataset("xsum", cache_dir=dataset_cache_dir)[split]

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        pair = self.dataset[index]
        return pair["document"], pair["summary"]

name_to_HFDS = {
    "cnn_dailymail": CNNDailyMailDataset,
    "xsum": XSumDataset
}

