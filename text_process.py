from typing import Callable, List, Tuple

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

from common import PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX, SPECIAL_SYMBOLS



# This class will process the raw text data and convert them into tensors
# and convert the tensors back to texts in the inference stage
# It contains:
# - vocabularies for source and target languages
# - tokenizers for source and target languages
class TextProcessor:
    SPACY_TOKENIZER_MODELS = {"de": "de_core_news_sm", "en": "en_core_web_sm"}

    def __init__(self, dataset, language_pair: Tuple[str]):
        # Tokenizer is used to split the sentence into words
        # str -> list[str]
        self._tokenizers = [
            get_tokenizer("spacy", self.SPACY_TOKENIZER_MODELS[language])
            for language in language_pair
        ]

        # Vocabularies (type: torchtext.vocab.vocab.Vocab) is used to convert the words into token ids
        # list[str] -> list[int]
        self._vocabularies = [
            self._create_vocabularies(dataset, i) for i in range(len(language_pair))
        ]

        # Sequentials is used to end-2-end convert the sentence into tensor
        # which is a sequence of functions
        # str -> tensor
        self._transform_sequentials = [
            self._create_sequentials(i) for i in range(len(language_pair))
        ]

    def _create_vocabularies(self, dataset, index: int):
        def _tokens_iterator():
            for data_sample in dataset:
                yield tokenizer(data_sample[index])

        tokenizer = self._tokenizers[index]
        vocab_transform = build_vocab_from_iterator(
            iterator=_tokens_iterator(),
            min_freq=1,  # build every work into vocabulary
            specials=SPECIAL_SYMBOLS,
            special_first=True,
        )
        # set the default index for unknown token
        vocab_transform.set_default_index(UNK_IDX)
        return vocab_transform

    # It will apply the transforms in order
    # And then convert the token ids into tensor
    # It applied to source and target in training
    # The original input is: sentence string: "I am a student."
    # The first transform is to split the string into words: ["I", "am", "a", "student", "."]
    # via functools.partial
    # The second transform is to convert the words into token ids: [12, 562, 33, 84, 15]
    # via torchtext.vocab.vocab.Vocab
    # The third transform is to add <bos> and <eos> to the token ids: [2, 12, 562, 33, 84, 15, 3]
    # via _tensor_transform
    @staticmethod
    def _sequentials(transforms: List[Callable]):
        def _func(x):
            for transform in transforms:
                x = transform(x)
            return x

        return _func

    @staticmethod
    def _tensor_transform(token_ids: List[int]) -> torch.Tensor:
        return torch.cat(
            (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
        )

    def _create_sequentials(self, index: int):
        return self._sequentials(
            [
                self._tokenizers[index],
                self._vocabularies[index],
                self._tensor_transform,
            ]
        )

    @property
    def vocab_sizes(self):
        return (len(x) for x in self._vocabularies)

    # convert the source sentence into tensor
    def to_tensor(self, sentence: str):
        return self._transform_sequentials[0](sentence)

    # convert the token ids into the target sentence
    # via vocabularies lookup via token ids
    def to_str(self, tokens):
        tokens = list(tokens.cpu().numpy())
        vocab = self._vocabularies[1]
        output = " ".join(vocab.lookup_tokens(tokens))
        # replace begin and end: <bos>, <eos>
        return output.replace(SPECIAL_SYMBOLS[2], "").replace(SPECIAL_SYMBOLS[3], "")

    # What is collate_fn:
    # merges a list of samples to form a mini-batch of Tensor(s).
    # Used when using batched loading from a map-style dataset.
    # It will convert batch of sentences into
    # matrix tensors which size is (batch_size, max_seq_len)
    # with padding value to 1
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self._transform_sequentials[0](src_sample.rstrip("\n")))
            tgt_batch.append(self._transform_sequentials[1](tgt_sample.rstrip("\n")))

        # just padding every sentence in a batch to the same length
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX).transpose(0, 1)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX).transpose(0, 1)
        return src_batch, tgt_batch
