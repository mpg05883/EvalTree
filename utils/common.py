# Names of the datasets we're interested in evaluating
DATASETS = [
    "DS-1000",
    "MATH",
    "MMLU",
]

# Map local dataset names to HuggingFace dataset identifiers
HF_DATASET_MAP = {
    "DS-1000": "xlangai/DS-1000",
    "MATH": "hendrycks/competition_math",
    "MMLU": "cais/mmlu",
}
