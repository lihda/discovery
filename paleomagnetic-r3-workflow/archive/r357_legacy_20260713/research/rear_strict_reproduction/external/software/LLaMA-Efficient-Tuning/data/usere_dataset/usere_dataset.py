import json
import datasets
from typing import Any, Dict, List
import logging

_DESCRIPTION = "An example of dataset for LLaMA."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
# _URL = "/mnt/wangyuhao/usere/training/nq-dev-2000.jsonl"
# _URL = "/mnt/wangyuhao/usere/training/nq-train-pure-qa-dense.jsonl"
# _URL = "/mnt/wangyuhao/usere/training/nq-train-pure-qa-dense-1pos-focus.jsonl"
_URL = "/mnt/wangyuhao/usere/training/nq-train-pure-qa-dense-1pos-unfocus.jsonl"


logger = logging.getLogger(__name__)


class UseReDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "output": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        logger.info(f"Getting data from URL: {_URL}")
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
        logger.info(f"Getting data from URL: {_URL}")
        for key, example in enumerate(example_dataset):
            yield key, example
