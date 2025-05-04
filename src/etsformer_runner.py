from runner_component import RunnerComponent
from src.ETSformer.utils.tools import set_seed
import random
import torch

from src.ETSformer.exp.exp_main import Exp_Main


class EtsformerRunner(RunnerComponent):
    def __init__(self, ts_attribute: str, args):
        super().__init__(ts_attribute, args)
        self.model = None
        self.params = None
        self.predictions = None
        self.metrics = None
        self.training_time = None
        self.prediction_time = None
        self.args = args
        self.args.seq_len = self.TRAINING_WINDOW
        self.args.pred_len = self.PREDICTION_WINDOW
        self.args.root_path = f"{self.MAIN_DIR}/{self.GROUP}/{self.AGGREGATION}/"
        self.args.data_path = f"{self.FILE_ID}.csv"
        self.args.model_id = "ETTh1"
        self.args.target = self.ts_attribute

    def run(self):
        id = random.randint(0, 10000)
        set_seed(id)
        # setting record of experiments
        setting = "{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_K{}_lr{}_{}_{}".format(
            self.args.model_id,
            self.args.model,
            self.args.data,
            self.args.features,
            self.args.seq_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.K,
            self.args.learning_rate,
            self.args.des,
            id,
        )

        Exp = Exp_Main
        exp = Exp(self.args)  # set experiments
        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        exp.train(setting)

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, data="val")
        exp.test(setting, data="test")

        self.create_record(exp.metrics, vars(self.args), exp.test_time, exp.train_time)

        torch.cuda.empty_cache()
