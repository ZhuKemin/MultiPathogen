# run_analysis.py

import subprocess
import os

def call_r_script(input_csv, output_dir="output"):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Call R script with arguments
    result = subprocess.run(
        ["Rscript", "analysis.R", input_csv, output_dir],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Successfully processed {input_csv}")
    else:
        print(f"Error processing {input_csv}: {result.stderr}")

# Example usage
file_list = [
    "data/raw/flu/sample_flu1.csv",
    "data/raw/flu/sample_flu2.csv",
    "data/raw/rsv/sample_rsv1.csv"
]

# Loop through files and call the R function
for file in file_list:
    call_r_script(file)
