from torchtext.datasets import multi30k, Multi30k

from torch.utils.data import DataLoader

from text_process import TextProcessor

# set the download and save info for multi30k
LOCAL_ROOT = "data/multi30k"
multi30k.URL["train"] = (
    "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
)
multi30k.URL["valid"] = (
    "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
)

language_pair = ("de", "en")
# use train data to build vocabularies
vocabularies_data = Multi30k(root=LOCAL_ROOT, split="train", language_pair=language_pair)

def load_data(batch_size: int):
    processor = TextProcessor(vocabularies_data, language_pair)

    train_loader = _create_data_loader(
        "train", batch_size, processor.collate_fn
    )

    validate_loader = _create_data_loader(
        "valid", batch_size, processor.collate_fn
    )

    return train_loader, validate_loader, processor


def create_preprocessor():
    processor = TextProcessor(vocabularies_data, language_pair)
    return processor


def _create_data_loader(
    split: str, batch_size: int, collate_fn: callable
):
    data = Multi30k(root=LOCAL_ROOT, split=split, language_pair=language_pair)
    return DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)
