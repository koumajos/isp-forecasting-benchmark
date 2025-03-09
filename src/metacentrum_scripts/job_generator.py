import os
import subprocess

path = ""
prediction_window = 1
training_window = 24
job_max_time_hours = 4


def create_job(model, file):
    with open(f"{path}/synced/temp_script.sh", "w") as f:
        command = f'python {path}/synced/src/main.py --main-dir "{path}/data" --aggregation "agg_1_hour" --file-id {file.split(".")[0]} --test-set-size 0.6 --valid-set-size 0.05 --training-window {training_window} --model-name "{model}" --impute "zeros" --prediction-window {prediction_window}'

        script = f"""
        #!/bin/bash
        #PBS -N {model}_{training_window}_{prediction_window}
        #PBS -l select=1:ncpus=4:mem=16gb
        #PBS -l walltime={job_max_time_hours}:00:00
    
        echo ${{PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}}
    
        # Load venv
        HOMEDIR={path}
        SYNCED=$HOMEDIR/synced
        HOSTNAME=`hostname -f`
        source $SYNCED/venv/bin/activate
    
        cd $SYNCED
    
    
        {command}
        """
        f.write(script)

    subprocess.run(["qsub", f"{path}/synced/temp_script.sh"])


for model in ["LSTM", "GRU", "LSTM_FCN", "GRU_FCN", "InceptionTime", "Resnet"]:
    for file in os.listdir(f"{path}/data/institutions/agg_1_hour/"):
        create_job(model, file)

    for file in os.listdir(f"{path}/data/ip_addresses_sample/agg_1_hour/"):
        create_job(model, file)

    for file in os.listdir(f"{path}/data/institution_subnets/agg_1_hour/"):
        create_job(model, file)
