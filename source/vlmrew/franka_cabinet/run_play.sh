#!/bin/bash

# Store the directory where the checkpoints are located
chkpnt_path="/home/levi/projects/IsaacLab/source/vlmrew/franka_cabinet/logs/2024-07-01_17-31-30"

# Store all the zip files in the directory in an array
zip_files=("$chkpnt_path"/*.zip)

# Loop through all the zip files
for zip_file in "${zip_files[@]}"
do
    # Run the python script with the checkpoint file
    echo "$zip_file"
    python play.py --checkpoint "$zip_file"
    
done

