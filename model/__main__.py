from model import app
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Neural network for binary classification")
    parser.add_argument("--test-file", type=str, help="provide the training dataset file")
    parser.add_argument("--train-file", type=str, help="provide the test dataset file")
    args = parser.parse_args()
    test_path = args.test_file
    train_path = args.train_file
    app.run(train_path,test_path)
