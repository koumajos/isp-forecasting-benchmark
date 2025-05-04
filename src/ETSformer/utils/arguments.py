import os

def add_etsformer_args(parser):
    parser.add_argument("--model_id", type=str, default="test", help="model id")
    parser.add_argument(
        "--model", type=str, default="ETSformer", help="model name, options: [ETSformer]"
    )

    # data loader
    parser.add_argument("--data", type=str, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./data/ETT/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="S",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument("--target", type=str, default=None, help="target feature in S or MS task")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=(
            f'/scratch/smoletim/job_{os.environ.get("PBS_JOBID")}' if os.environ.get("PBS_JOBID") else "./checkpoints/"
        ),
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=0, help="start token length")
    parser.add_argument("--pred_len", type=int, help="prediction sequence length")

    # model define
    parser.add_argument("--enc_in", type=int, default=1, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=1, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=1, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=2, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--K", type=int, default=2, help="Top-K Fourier bases")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument(
        "--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]"
    )
    parser.add_argument("--activation", type=str, default="sigmoid", help="activation")

    parser.add_argument("--min_lr", type=float, default=1e-30)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--std", type=float, default=0.2)

    parser.add_argument("--smoothing_learning_rate", type=float, default=0, help="optimizer learning rate")
    parser.add_argument("--damping_learning_rate", type=float, default=0, help="optimizer learning rate")
    parser.add_argument("--output_attention", type=bool, default=False)

    # optimization
    parser.add_argument("--optim", type=str, default="adam", help="optimizer")
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=15, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="Exp", help="exp description")
    parser.add_argument("--lradj", type=str, default="exponential_with_warmup", help="adjust learning rate")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=False, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    return parser
