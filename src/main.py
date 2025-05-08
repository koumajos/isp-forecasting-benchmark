import argparse

from src.dl_runner import DLRunner
from src.rclstm_runner import RclstmRunner
from src.lstmae_runner import LstmaeRunner
from src.etsformer_runner import EtsformerRunner
from src.ETSformer.utils.arguments import add_etsformer_args

parser = argparse.ArgumentParser(description="Argument parser for data processing parameters.")

parser.add_argument("--main-dir", type=str, default=".", help="Main directory for data.")
parser.add_argument(
    "--aggregation",
    type=str,
    default="agg_1_hour",
    choices=["agg_1_day", "agg_1_hour", "agg_10_minutes"],
    help="Aggregation method.",
)
parser.add_argument("--file-id", type=int, default=1, help="File ID.")
parser.add_argument(
    "--subnets",
    action="store_true",
    default=False,
    help="Flag to indicate if subnets should be used.",
)
parser.add_argument(
    "--sample",
    action="store_true",
    default=False,
    help="Flag to indicate if sample dataset should be used.",
)
parser.add_argument("--test-set-size", type=float, default=0.6, help="Test set size.")
parser.add_argument("--valid-set-size", type=float, default=0.05, help="Validation set size.")
parser.add_argument("--training-window", type=int, default=24, help="Window size.")
parser.add_argument(
    "--ts-attributes",
    nargs="+",
    default=[],
    help="Time series attributes to use for training.",
)
parser.add_argument("--model-name", type=str, default="GRU_FCN", help="Model name.")
parser.add_argument("--impute", type=str, default="zeros", help="Imputation method.")
parser.add_argument("--prediction-window", type=int, default=1, help="Number of samples to predict.")
parser.add_argument("--anomaly-threshold", type=float, default=0.09, help="Value for anomaly threshold.")
parser.add_argument("--plot-anomalies", action="store_true", default=False, help="Flag to plot anomalies.")

args, remaining = parser.parse_known_args()
if args.model_name == "ETSformer":
    parser = add_etsformer_args(parser)

args = parser.parse_args()

if args.sample and args.aggregation == "agg_10_minutes":
    all_ts_attributes = [
        "n_flows",
        "n_packets",
        "n_bytes",
        "n_dest_asn",
        "n_dest_ports",
        "n_dest_ip",
        "tcp_udp_ratio_packets",
        "tcp_udp_ratio_bytes",
        "dir_ratio_packets",
        "dir_ratio_bytes",
        "avg_duration",
        "avg_ttl",
    ]
else:
    all_ts_attributes = [
        "n_flows",
        "n_packets",
        "n_bytes",
        "sum_n_dest_asn",
        "average_n_dest_asn",
        "std_n_dest_asn",
        "sum_n_dest_ports",
        "average_n_dest_ports",
        "std_n_dest_ports",
        "sum_n_dest_ip",
        "average_n_dest_ip",
        "std_n_dest_ip",
        "tcp_udp_ratio_packets",
        "tcp_udp_ratio_bytes",
        "dir_ratio_packets",
        "dir_ratio_bytes",
        "avg_duration",
        "avg_ttl",
    ]

if __name__ == "__main__":
    print(args)
    attrs = all_ts_attributes if not args.ts_attributes else args.ts_attributes
    for attr in attrs:
        if args.model_name == "RCLSTM":
            runner = RclstmRunner(attr, args)
        elif args.model_name == "LSTMAE":
            runner = LstmaeRunner(attr, args)
        elif args.model_name == "ETSformer":
            runner = EtsformerRunner(attr, args)
        else:
            runner = DLRunner(attr, args)
        runner.run()
