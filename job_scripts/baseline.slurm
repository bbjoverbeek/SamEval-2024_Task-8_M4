#!/bin/bash

# Script used to train and evaluate a baseline.
# More info about parameters can be found here:
# https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system

# Set name of the job
#SBATCH --job-name=Shared_Task_Baseline

# Set time limit of 6 hours (will be cancelled after that)
#SBATCH --time=6:00:00

# Set values for computing power
#SBATCH --nodes=1               # default = 1
#SBATCH --cpus-per-task=1       # default = 1
#SBATCH --mem-per-cpu=12GB      # amount of memory per cpu

# Set values for using GPU
#SBATCH --gpus-per-node=a100:2

# Command for loading Python on Hábrók
module purge
module load Python/3.9.6-GCCcore-11.2.0

# Command for (creating and) activating a venv and installing required Python packages
if [ ! -d "env" ]; then
  python3 -m venv env
fi

source env/bin/activate

pip install -r requirements.txt | grep -v 'already satisfied'

# program variables
model="distilbert-base-cased"
task="A"
predictions_file="predictions/${model}_subtask${task}_monolingual.txt"

# Command for running task
python3 baseline.py \
  -train_file_path data/Subtask${task}/subtaskA_train_monolingual.jsonl \
  -test_file_path data/Subtask${task}/subtaskA_dev_monolingual.jsonl \
  -subtask $task \
  -model $model \
  -prediction_file_path $predictions_file

# extract scores from prediction file

# notify via ntfy that the model is done
curl \
  -H "Title: ${model} is done training on Hábrók" \
  -H "Priority: urgent" \
  -d "These are the scores: 123" \
  ntfy.sh/habrok_model_done

# remove model folder
rm -r $model