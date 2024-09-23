#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --account=default
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --job-name=cube-pdb
#SBATCH --output=eureka/output.txt

module load gcc
module load glfw/3.3.8
module load glew/2.2.0


echo "Compiling the project..."
make clean
make


if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    OUTPUT_FILE="bin/main"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Running the executable..."
        $OUTPUT_FILE BatchIDA 50 11  # Run the executable
    else
        echo "Error: Output file not found!"
    fi
else
    echo "Compilation failed!"
fi