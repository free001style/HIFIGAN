import csv
import json
import os
import shutil
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class LJSpeechDataset(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "LJSpeech-1.1"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        index = self._get_or_load_index()

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        with open(self._data_dir / "metadata.csv", "r") as csv_file:
            csv_file_iter = csv.reader(csv_file, delimiter="|", quotechar="|")
            for file_id, _, _ in csv_file_iter:
                path = str(self._data_dir / "wavs" / f"{file_id}.wav")
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                index.append({"audio_path": path, "audio_len": length})
        return index
