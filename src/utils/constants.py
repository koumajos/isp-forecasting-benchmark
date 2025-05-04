from tsai.all import GRU, GRU_FCN, LSTM, LSTM_FCN, Adam, InceptionTime, ResNet

MODEL_MAPPER = {
    "LSTM_FCN": LSTM_FCN,
    "GRU_FCN": GRU_FCN,
    "InceptionTime": InceptionTime,
    "Resnet": ResNet,
    "LSTM": LSTM,
    "GRU": GRU,
}

base_lstmae_config = {
    "batch_size": 32,
    "num_layers": 1,
    "epochs": 100,
    "lr": 0.01,
    "patience": 10,
}

MODEL_PARAMS = {
    "LSTM": {
        "batch_size": 16,
        "bidirectional": True,
        "epochs": 100,
        "hidden_size": 100,
        "lr": 0.01,
        "optimizer": Adam,
        "patience": 5,
        "n_layers": 1,
    },
    "LSTM_FCN": {
        "batch_size": 16,
        "bidirectional": True,
        "conv_layers": (128, 256, 128),
        "epochs": 100,
        "fc_dropout": 0.5,
        "hidden_size": 100,
        "kss": (3, 3, 3),
        "lr": 0.01,
        "optimizer": Adam,
        "patience": 5,
        "rnn_dropout": 0.5,
        "rnn_layers": 1,
    },
    "GRU": {
        "batch_size": 16,
        "bidirectional": True,
        "epochs": 100,
        "hidden_size": 100,
        "lr": 0.01,
        "optimizer": Adam,
        "patience": 5,
        "n_layers": 1,
    },
    "GRU_FCN": {
        "batch_size": 16,
        "bidirectional": True,
        "conv_layers": (64, 128, 64),
        "epochs": 20,
        "fc_dropout": 0.5,
        "hidden_size": 20,
        "kss": (7, 5, 3),
        "lr": 0.01,
        "optimizer": Adam,
        "patience": 5,
        "rnn_dropout": 0.3,
        "rnn_layers": 4,
    },
    "InceptionTime": {
        "batch_size": 128,
        "epochs": 20,
        "ks": 20,
        "lr": 0.001,
        "nf": 16,
        "optimizer": Adam,
        "patience": 5,
    },
    "Resnet": {
        "batch_size": 128,
        "epochs": 20,
        "lr": 0.01,
        "optimizer": Adam,
        "patience": 5,
    },
    "RCLSTM": {
        "connectivity": 0.01,
        "hidden_size": 300,
        "batch_size": 32,
        "dropout": 0.5,
        "num_layers": 1,
        "epochs": 100,
    },
    "LSTMAE": {
        24: {
            **base_lstmae_config,
            "hidden_size": 16,
        },
        168: {
            **base_lstmae_config,
            "hidden_size": 112,
        },
    },
}

DATA_GROUPS = ["institution_subnets", "institutions", "ip_addresses_sample"]
