import argparse
from dotenv import load_dotenv

from model import TranslationModel
from data import create_preprocessor, load_data


def train():
    # The traning epochs, for my tesing, 50 is a good epochs
    epochs = 50
    batch_size = 128

    train_loader, validate_loader, preprocessor = load_data(batch_size)
    model = TranslationModel(preprocessor)
    # model.load()

    model.train(train_loader, validate_loader, epochs)


def infer():
    preprocessor = create_preprocessor()
    model = TranslationModel(preprocessor)
    # model.load("translation_transform_model_best.pt")
    model.load()

    sentence = "Eine Gruppe von Menschen steht vor einem Iglu."
    translation = model.infer(sentence)
    print(f"Source: {sentence}, Translation({len(translation)}): {translation}")


def upload():
    preprocessor = create_preprocessor()
    model = TranslationModel(preprocessor)
    model.upload()


def from_pretrain():
    preprocessor = create_preprocessor()
    model = TranslationModel(preprocessor)
    model.from_pretrain()
    sentence = "Eine Gruppe von Menschen steht vor einem Iglu."
    translation = model.infer(sentence)
    print(f"Source: {sentence}, Translation({len(translation)}): {translation}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run MNIST generation model with different modes: train/infer/upload/from_pretrain"
    )
    parser.add_argument(
        "mode",
        choices=["train", "infer", "upload", "from_pretrain"],
        help="The mode to run",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()
        infer()
    elif args.mode == "infer":
        infer()
    elif args.mode == "upload":
        upload()
    elif args.mode == "from_pretrain":
        from_pretrain()
    else:
        print(f"The mode={args.mode} not supported")
