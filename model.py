# run the following command to download the spacy models
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from huggingface_hub import upload_file, create_repo, hf_hub_download


from module import Seq2SeqTransformer
from mask import create_all_masks, generate_square_subsequent_mask
from common import PAD_IDX, BOS_IDX, EOS_IDX


class TranslationModel:
    def __init__(self, preprocessor) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # define model structure
        self.embedding_dim = 512
        # number of attention heads in the transformer
        nhead = 8
        # number of transformer layers
        num_layers = 2
        # the hidden dim of the feedforward layer in the transformer
        feedforward_dim = 256
        # dropout probability
        dropout = 0.4

        # for multi30k: source_vocab_size=19214 target_vocab_size=10837
        source_vocab_size, target_vocab_size = preprocessor.vocab_sizes
        # define model
        model = Seq2SeqTransformer(
            self.embedding_dim,
            source_vocab_size,
            target_vocab_size,
            nhead,
            num_layers,
            feedforward_dim,
            dropout,
        ).to(self.device)
        self.model = self._init_parameters(model)

        self.model_fn = "multi30k_de_en_transformer_model.pt"
        self.hf_modle_fn = "multi30k_de_en_transformer"
        self._preprocessor = preprocessor

    def _init_parameters(self, model):
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model.to(self.device)

    def train(self, train_loader, validate_loader, epochs: int = 10, lr=1e-4):
        # Define the loss function, which ignores the index of padding token
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        # optimizer, training hyperparameters here
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
        )
        best_val_loss = float("inf")
        for epoch in range(epochs):
            # Train the model
            train_loss = self.train_epoch(
                optimizer, criterion, train_loader, f"Epoch {epoch+1}/{epochs} Train"
            )

            val_loss = self.evaluate(validate_loader, criterion, f"Epoch {epoch+1}/{epochs} Eval")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_fn)

            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    def train_epoch(self, optimizer, criterion, train_loader, desc):
        self.model.train()
        train_loss = 0.0

        total = 0
        for source, target in tqdm(train_loader, desc=desc):
            # which shape is (batch_size, seq_len)
            # seq_len is various, batch_size is fixed
            src = source.to(self.device)
            tgt = target.to(self.device)

            # the model will recurrent infer the target sequence
            # so the target sequence should be shifted by one in training
            # what ths loss it is
            tgt_in = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_all_masks(
                src, tgt_in, self.device, PAD_IDX
            )

            predicts = self.model(
                src,
                tgt_in,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )
            optimizer.zero_grad()

            # the model will recurrent infer the target sequence
            # so the target sequence should be shifted by one
            # to make the loss function to compare the predicted target sequence
            tgt_out = tgt[:, 1:]
            # which predicts shape is (batch_size, tgt_seq_len - 1, target_vocab_size)
            # while tgt_out shape is (batch_size, tgt_seq_len - 1)
            # flatten logits and target by removing the batch dimension
            loss = criterion(predicts.reshape(-1, predicts.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += 1
        return train_loss / total

    def evaluate(self, validate_loader, criterion, desc):
        self.model.eval()
        val_loss = 0.0
        total = 0
        for source, target in tqdm(validate_loader, desc=desc):
            src = source.to(self.device)
            tgt = target.to(self.device)

            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_all_masks(
                src, tgt_input, self.device, PAD_IDX
            )

            predicts = self.model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            tgt_out = tgt[:, 1:]
            loss = criterion(predicts.reshape(-1, predicts.shape[-1]), tgt_out.reshape(-1))

            val_loss += loss.item()
            total += 1

        return val_loss / total

    def load(self, model_fn: str = None):
        model_fn = model_fn or self.model_fn
        self.model.load_state_dict(torch.load(model_fn))

    def infer(self, sentence: str) -> str:
        self.model.eval()
        # process source
        src = self._preprocessor.to_tensor(sentence)
        src = src.unsqueeze(0).to(self.device)

        num_tokens = src.shape[1]
        # 5 is a priori assumption of the max length of the target sentence
        # which may vary in different languages
        tgt_tokens = self.greedy_decode(
            src, max_len=num_tokens + 5, start_symbol=BOS_IDX
        ).flatten()
        return self._preprocessor.to_str(tgt_tokens)

    def greedy_decode(self, src, max_len, start_symbol):
        # the transformer encoder part which will output the memory
        # the encoded Key/Value for the next decoder stage
        # which shape is (source_seq_len, 1, embedding_dim)
        memory = self.model.encode(src)
        # pylint: disable=no-member
        # The target tokens
        # First we initialize it to be the index of start symbols <bos>, which
        # used as the previous tokens for the next tokens
        # then we decode(infer) the future tokens one by one
        target = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for _ in range(max_len - 1):
            # tgt_mask, which is a triangle matrix, is used to mask out the future tokens
            tgt_mask = generate_square_subsequent_mask(target.size(1), self.device)
            next_token = self.model.decode(target, memory, tgt_mask)
            target = torch.cat([target, next_token], dim=1).to(self.device)
            if next_token == EOS_IDX:
                break
        return target

    def upload(self, model_fn: str = None):
        model_fn = model_fn or self.model_fn
        token = os.getenv("HUGGINGFACE_TOKEN")
        repo_id = os.getenv("HUGGINGFACE_REPO")
        create_repo(
            repo_id,
            token=token,
            private=False,
            repo_type="model",
            exist_ok=True,
        )

        upload_file(
            repo_id=repo_id,
            path_or_fileobj=model_fn,
            path_in_repo=self.hf_modle_fn,
            token=token,
        )

    def from_pretrain(self):
        repo_id = os.getenv("HUGGINGFACE_REPO")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=self.hf_modle_fn,
            cache_dir="./cache",
        )
        # model_path = try_to_load_from_cache(repo_id=repo_id, filename=self.hf_modle_fn)
        self.load(model_path)
