# src/data/datasets/__init__.py
#
# Dataset registry for SFT.
# Each entry is a dedicated loader file for one dataset source.
# To add a new dataset: create src/data/datasets/mydata.py with a load() function,
# then add it to _REGISTRY below.

from .alpaca import load as _load_alpaca
from .dolly import load as _load_dolly
from .openorca import load as _load_openorca
from .local import load as _load_local

_REGISTRY = {
    "alpaca":   _load_alpaca,    # tatsu-lab/alpaca          — 52k examples
    "dolly":    _load_dolly,     # databricks-dolly-15k      — 15k examples
    "openorca": _load_openorca,  # Open-Orca/OpenOrca        — ~1M examples
    "local":    _load_local,     # local JSONL/JSON file
}


def available_datasets() -> list[str]:
    return list(_REGISTRY.keys())


def load_datasets(
    names: list[str],
    local_path: str = None,
    max_samples_per_dataset: int = None,
) -> list[dict]:
    """
    Load and combine one or more datasets by name. Shuffling happens in the caller.

    Args:
        names:                   List of dataset names from the registry.
        local_path:              Path to local JSONL/JSON file (required when 'local' is in names).
        max_samples_per_dataset: Cap examples per dataset (useful for balancing large + small sources).

    Returns:
        Combined list of {"instruction", "input", "output"} dicts.

    Examples:
        load_datasets(["alpaca"])
        load_datasets(["alpaca", "dolly"])
        load_datasets(["local"], local_path="data/sft_data.jsonl")
        load_datasets(["alpaca", "dolly", "openorca"], max_samples_per_dataset=5000)
    """
    all_examples = []

    for name in names:
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {available_datasets()}\n"
                "To add a new source, create src/data/datasets/<name>.py with a load() function."
            )

        print(f"\n[Dataset] Loading '{name}'...")
        loader = _REGISTRY[name]

        if name == "local":
            if not local_path:
                raise ValueError(
                    "'local' dataset requires a path. "
                    "Pass local_path= or use --local-data-path in the CLI."
                )
            examples = loader(path=local_path, max_samples=max_samples_per_dataset)
        else:
            examples = loader(max_samples=max_samples_per_dataset)

        print(f"[Dataset] '{name}': {len(examples)} examples")
        all_examples.extend(examples)

    print(f"\n[Dataset] Total combined: {len(all_examples)} examples from {names}")
    return all_examples
