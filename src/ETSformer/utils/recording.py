import pandas as pd
from datetime import datetime
import os


def create_record(
    metrics,
    params,
    prediction_time,
    training_time,
    model_name,
    patience,
    column,
    main_dir,
    aggregation,
    group,
    file_id,
    PBS_JOB_ID,
):
    params["timestamp"] = datetime.now()
    record = pd.DataFrame(
        [
            {
                "metrics": metrics,
                "Training Time": training_time,
                "Prediction Time": prediction_time,
                "params": params,
                "file path": f"{main_dir}/output/{group}/{aggregation}/transformer/ETSformer/{file_id}.csv",
                "timestamp": datetime.now(),
                "model_name": model_name,
                "patience": patience,
                "column": column,
                "file_id": file_id,
            }
        ]
    )
    try:
        record.to_csv(
            f"{main_dir}/output/records/{group}/{aggregation}/transformer/ETSformer/record_{file_id}_{PBS_JOB_ID}_{column}.csv"
        )
    except OSError:
        os.makedirs(f"{main_dir}/output/records/{group}/{aggregation}/transformer/ETSformer")
        record.to_csv(
            f"{main_dir}/output/records/{group}/{aggregation}/transformer/ETSformer/record_{file_id}_{PBS_JOB_ID}_{column}.csv"
        )
