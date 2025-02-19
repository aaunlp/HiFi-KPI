import argparse
from huggingface_hub import hf_hub_download
import json
import pandas as pd
from datasets import load_dataset

from GranularitySelection import multi_level_collapse


parser = argparse.ArgumentParser(description="Choose taxonomy and number of iterations.")
parser.add_argument(
    "--taxonomy",
    choices=["calculation", "presentation"],
    default="presentation",
    help="Select the taxonomy to use (default: presentation)."
)
parser.add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Number of iterations for multi_level_collapse (default: 1)."
)
args = parser.parse_args()


dataset = load_dataset("AAU-NLP/HiFi-KPI")
train = dataset["train"]   
validation = dataset["validation"]
test = dataset["test"]

# Determine the taxonomy file name based on the argument
taxonomy_file = (
    "calculationMasterTaxonomy.json"
    if args.taxonomy == "calculation"
    else "presentationMasterTaxonomy.json"
)

file_path = hf_hub_download(
    repo_id="AAU-NLP/HiFi-KPI", 
    filename=taxonomy_file, 
    repo_type="dataset" 
)

with open(file_path, "r") as f:
    taxonomy = json.load(f)

taxonomy_df = pd.DataFrame(taxonomy)
print("started granularity selection.")
# Process the data using multi_level_collapse
train, validation, test = multi_level_collapse(taxonomy_df, train, validation, test, args.iterations)


# Define output file names with the iteration count in them
output_prefix = f"Granularity{args.iterations}"
train_output_path = f"{output_prefix}_train.json"
validation_output_path = f"{output_prefix}_validation.json"
test_output_path = f"{output_prefix}_test.json"

# Save the processed data back to JSON files
with open(train_output_path, 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=4)
with open(validation_output_path, 'w', encoding='utf-8') as f:
    json.dump(validation, f, ensure_ascii=False, indent=4)
with open(test_output_path, 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=4)
