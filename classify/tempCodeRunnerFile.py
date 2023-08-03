

    train_tfms = tt.Compose(
        [tt.Resize((28, 28)), tt.ToTensor(), tt.Normalize(mean, std)]
    )

    test_tfms = tt.Compose([tt.Resize((28, 28)), tt.ToTensor()])