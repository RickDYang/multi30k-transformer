# Transformer for MULTI30K de-2-en translation
This repositry is for AI beginers to learn transformer model for NLP from code scratch.

Currently, the AI models are becoming more and more complicated, and cost more and more computing resourse. It is getting harder for AI beginer to learn. I would like to implement the foundation models from minimal requirements.
My GPU is **NVIDIA GeForce RTX 2070**, which has 8Gb GPU Memory. I try to implement the models based on this minimal GPU requirements.

For this model is for the translation task from Germany to English, and is trained on MUTII30K data.

It is directly use torch.nn.Transformer to build the model, so the code will not demostrate the detailed implementation of Transformer model. Instead, it focuses on tokenizer/vocabulary/mask/padding etc.

# References
Firstly, I would like to recommend to read the following articles to understand what Transformer model is.

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)


# Setup
- [Setup torch with CUDA](https://pytorch.org/get-started/locally/)
- Setup torchtext with CUDA
    ```
    conda install  torchtext==0.18.0  pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
- Clone the repo
- Setup requirements.txt
    ```shell
    pip install -r requirements.txt
    ```
- download spacy models
    We use spacy tokenizers so please download the spacy models via
    ```shell
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    ```

# How to run
The code is running in 4 modes: train, infer, upload and from_pretrain. Use the following command to run:
```shell
python main.py [train, infer, upload, from_pretrain]
```
## Environment Variables
Before running upload/from_pretrain, you need to set the following environment variables for huggingface access.
```
HUGGINGFACE_REPO=username/reponame
HUGGINGFACE_TOKEN=write_token_to_upload
```
## Modes
- train

    During train, the model will be saved to "mnist_vae_model.pt" for the best model
- infer

    In the inferencing, the model will be loaded from "mnist_vae_model.pt"
- upload

    You can use upload to upload the model to huggingface hub
- from_pretrain

    It will download the model previously uploaded from huggingface hub and make inference.

    My trained models is in [Here](https://huggingface.co/RickDYang/ai-mini/blob/main/multi30k_de_en_transformer)