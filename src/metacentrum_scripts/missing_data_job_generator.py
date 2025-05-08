import subprocess
import argparse

parser = argparse.ArgumentParser(description="Argument parser for input file")

parser.add_argument("--missing-file", type=str, required=True, help="Main directory for data.")

args = parser.parse_args()
missing_file = args.missing_file
job_name = missing_file.split("/")[-1][:-4]

"""
Use for creating jobs generated based on output of find_missing_combinations.ipynb.
"""

path = ""
hours_needed = 4
jobs_at_once = 8  # number of jobs to compute within one job

i = 0
with open(missing_file, "r") as commands:
    commands_lines = commands.readlines()
    for line in range(0, len(commands_lines), jobs_at_once):
        with open(f"{path}/synced/temp_script.sh", "w") as f:
            command = "\n".join(commands_lines[line : line + jobs_at_once])

            script = f"""
            #!/bin/bash
            #PBS -N {job_name}
            #PBS -l select=1:ncpus=1:mem=4gb
            #PBS -l walltime={hours_needed}:00:00


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
