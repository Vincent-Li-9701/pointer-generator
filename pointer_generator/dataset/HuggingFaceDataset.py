from datasets import load_dataset
from pointer_generator.utils.config import dataset_cache_dir


class HuggingFaceDataset(object):
    def __init__(self, dataset_name, additional_config=None, split="train", doc_name="document", summary_name="summary"):
        if additional_config is not None:
            self.dataset = load_dataset(dataset_name, additional_config, cache_dir=dataset_cache_dir, split=split)
        else:
            self.dataset = load_dataset(dataset_name, cache_dir=dataset_cache_dir, split=split)
        self.doc_name = doc_name
        self.summary_name = summary_name

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        pair = self.dataset[index]
        return pair[self.doc_name], pair[self.summary_name]


class CNNDailyMailDataset(HuggingFaceDataset):
    def __init__(self, split="train", version='3.0.0'):
        assert split in ["train", "test", "validation"]
        assert version in ["3.0.0", "2.0.0", "1.0.0"]
        super(CNNDailyMailDataset, self).__init__("cnn_dailymail", additional_config=version, split=split, doc_name="article", summary_name="highlights")


class XSumDataset(HuggingFaceDataset):
    def __init__(self, split="train"):
        assert split in ["train", "test", "validation"]
        super(XSumDataset, self).__init__("xsum", split=split)


class MultiNewsDataset(HuggingFaceDataset):
    def __init__(self, split="train"):
        assert split in ["train", "test", "validation"]
        super(MultiNewsDataset, self).__init__("multi_news", split=split)


class GigaWordDataset(HuggingFaceDataset):
    def __init__(self, split="train"):
        assert split in ["train", "test", "validation"]
        super(GigaWordDataset, self).__init__("gigaword", split=split)


class RedditTIFUDataset(HuggingFaceDataset):
    def __init__(self, version="short", split="train", summary_name="tldr"):
        assert version in ["long", "short"]
        # tldr vs title
        assert summary_name in ["tldr", "title"]
        super(RedditTIFUDataset, self).__init__("reddit_tifu", additional_config=version, doc_name="documents", summary_name=summary_name)


class AESLCDataset(HuggingFaceDataset):
    def __init__(self, split="train"):
        assert split in ["train", "test", "validation"]
        super(AESLCDataset, self).__init__("aeslc", split=split, doc_name="email_body", summary_name="subject_line")


class BillSumDataset(HuggingFaceDataset):
    def __init__(self, split="train", summary_name="title"):
        if split == "validation":
            split = "ca_test"
        assert split in ["train", "test", "ca_test"]
        # summary (long) vs title (short)
        assert summary_name in ["summary", "title"]
        super(BillSumDataset, self).__init__("billsum", split=split, doc_name="text", summary_name=summary_name)


# Manual: WikiHow, NewsRoom
# class WikiHowDataset(object):
#     def __init__(self, split="train"):
#         dataset = load_dataset("wikihow", "all", cache_dir=dataset_cache_dir)
#         dataset = load_dataset("newsroom", cache_dir=dataset_cache_dir)


name_to_HFDS = {
    "cnn_dailymail": CNNDailyMailDataset,
    "xsum": XSumDataset,
    "multi_news": MultiNewsDataset,
    "gigaword": GigaWordDataset,
    # "reddit_tifu": RedditTIFUDataset,
    # "aeslc": AESLCDataset,
    "billsum": BillSumDataset
}

