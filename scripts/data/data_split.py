import json

import datasets
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

SPLIT_SEED = 42

source_repo = "uoft-cs/cifar100"
output_repo = "mlkbsalah/cifar100"

api = HfApi()
source_revision = api.dataset_info(source_repo).sha

source = load_dataset(source_repo, revision=source_revision)

split = source["train"].train_test_split(test_size=0.1, seed=SPLIT_SEED, stratify_by_column="fine_label")

data = DatasetDict({
    "train": split["train"],
    "val": split["test"],
    "test": source["test"],
})

metadata = {
    "source_repo": source_repo,
    "source_revision": source_revision,
    "seed": SPLIT_SEED,
    "validation_fraction": 0.1,
    "stratify_by_column": "fine_label",
    "versions": {
        "datasets": datasets.__version__
    }
}

data.push_to_hub(output_repo, private=True)

api.upload_file(
    path_or_fileobj=json.dumps(metadata, indent=2).encode("utf-8"),
    path_in_repo="split_metadata.json",
    repo_id=output_repo,
    repo_type="dataset",
)
