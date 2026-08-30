from .alpaca   import load as _load_alpaca
from .dolly    import load as _load_dolly
from .openorca import load as _load_openorca
from .local    import load as _load_local

_REGISTRY = {
    "alpaca":   _load_alpaca,
    "dolly":    _load_dolly,
    "openorca": _load_openorca,
    "local":    _load_local,
}

def available_datasets() -> list[str]:
    return list(_REGISTRY.keys())

def load_datasets(
    names: list[str],
    local_path: str = None,
    max_samples_per_dataset: int = None,
) -> list[dict]:
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
