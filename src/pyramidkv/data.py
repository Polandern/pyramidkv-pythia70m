from __future__ import annotations

from pathlib import Path


def load_text(
    sample_text: str | None = None,
    dataset: str | None = None,
    dataset_config: str | None = None,
    split: str = "test",
    max_documents: int = 1,
    streaming: bool = False,
) -> str:
    if sample_text:
        return Path(sample_text).read_text(encoding="utf-8")

    if not dataset:
        return Path("samples/tiny_pg19.txt").read_text(encoding="utf-8")

    from datasets import load_dataset

    kwargs = {}
    if dataset_config:
        kwargs["name"] = dataset_config
    ds = load_dataset(dataset, **kwargs, split=split, streaming=streaming)
    texts = []
    if streaming:
        rows = iter(ds)
    else:
        rows = iter(ds.select(range(min(max_documents, len(ds)))))
    for row_idx, row in enumerate(rows):
        if row_idx >= max_documents:
            break
        if "text" in row and row["text"]:
            texts.append(row["text"])
    return "\n\n".join(texts)
