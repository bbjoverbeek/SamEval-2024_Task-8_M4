#!/bin/bash

# load modules
echo "-> Purging Modules"
module purge

echo "-> Loading Python3.11"
module load Python/3.11.3-GCCcore-12.3.0
echo -n "\$python3 --version: "
python3 --version

# Command for (creating and) activating a venv and installing required Python packages
if [ ! -d "env" ]; then
    echo "-> Creating new venv"
    python3 -m venv env
fi

echo "-> Activating venv"
source env/bin/activate

echo "-> Installing requirements.txt"
pip install -r requirements.txt | grep -v 'already satisfied'

# running the script
echo "-> Running run.py"
python3 run.py