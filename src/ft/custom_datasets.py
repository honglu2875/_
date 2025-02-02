from datasets import load_dataset

from ft.dataset_config import DatasetConfig, DatasetMixConfig, DatasetType

DATASETS = {
    "allenai/c4": DatasetConfig(
        type=DatasetType.PRETRAIN,
        path="allenai/c4",
        loader=lambda path: load_dataset(path, name="en", split="train", streaming=True),
        text_processor=lambda x: x["text"],
    ),
    "ft-reasoning0": DatasetMixConfig(
        datasets=[
            DatasetConfig(
                type=DatasetType.REASONING,
                path="Idavidrein/gpqa",
                loader=lambda path: load_dataset(path, name="gpqa_extended"),
                text_processor=lambda x: x["Question"],
                extra_processor=lambda x: x["Correct Answer"],
            ),
            DatasetConfig(
                type=DatasetType.REASONING,
                path="HuggingFaceH4/MATH-500",
                loader=lambda path: load_dataset(path),
                text_processor=lambda x: x["problem"],
                extra_processor=lambda x: x["answer"],
            ),
            DatasetConfig(
                type=DatasetType.REASONING,
                path="cruxeval-org/cruxeval",
                loader=lambda path: load_dataset(path),
                text_processor=lambda x: f"Given the following code:\n```\n{x['code']}```\nWith the input being {x['input']}, what is the output?",  # noqa
                extra_processor=lambda x: x["output"],
            ),
        ]
    ),
}
