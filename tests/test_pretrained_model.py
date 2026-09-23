import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from graphormer.pretrain import load_pretrained_model


class LoadPretrainedModelTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.model_state = {"encoder.weight": torch.tensor([1.0, 2.0])}

    def test_loads_fairseq_checkpoint(self):
        checkpoint_path = self.directory / "checkpoint.pt"
        torch.save({"model": self.model_state, "extra_state": {}}, checkpoint_path)

        loaded = load_pretrained_model(
            pretrained_model_path=str(checkpoint_path)
        )

        torch.testing.assert_close(
            loaded["encoder.weight"],
            self.model_state["encoder.weight"],
        )

    def test_loads_raw_state_dictionary(self):
        checkpoint_path = self.directory / "state_dict.pt"
        torch.save(self.model_state, checkpoint_path)

        loaded = load_pretrained_model(
            pretrained_model_path=str(checkpoint_path)
        )

        torch.testing.assert_close(
            loaded["encoder.weight"],
            self.model_state["encoder.weight"],
        )

    def test_rejects_missing_path(self):
        checkpoint_path = self.directory / "missing.pt"

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            load_pretrained_model(pretrained_model_path=str(checkpoint_path))

    def test_rejects_directory_path(self):
        with self.assertRaisesRegex(IsADirectoryError, "is not a file"):
            load_pretrained_model(pretrained_model_path=str(self.directory))

    def test_rejects_malformed_checkpoint(self):
        checkpoint_path = self.directory / "malformed.pt"
        torch.save(["not", "a", "state", "dictionary"], checkpoint_path)

        with self.assertRaisesRegex(ValueError, "state dictionary"):
            load_pretrained_model(pretrained_model_path=str(checkpoint_path))

    def test_rejects_name_and_path_together(self):
        checkpoint_path = self.directory / "checkpoint.pt"
        torch.save(self.model_state, checkpoint_path)

        with self.assertRaisesRegex(ValueError, "not both"):
            load_pretrained_model(
                pretrained_model_name="pcqm4mv2_graphormer_base",
                pretrained_model_path=str(checkpoint_path),
            )

    def test_named_model_download_is_unchanged(self):
        with mock.patch(
            "graphormer.pretrain.dist.is_initialized",
            return_value=False,
        ), mock.patch(
            "graphormer.pretrain.load_state_dict_from_url",
            return_value={"model": self.model_state},
        ) as download:
            loaded = load_pretrained_model("pcqm4mv2_graphormer_base")

        download.assert_called_once()
        torch.testing.assert_close(
            loaded["encoder.weight"],
            self.model_state["encoder.weight"],
        )

    def test_local_model_loads_when_distributed_is_initialized(self):
        checkpoint_path = self.directory / "checkpoint.pt"
        torch.save({"model": self.model_state}, checkpoint_path)

        with mock.patch(
            "graphormer.pretrain.dist.is_initialized",
            return_value=True,
        ), mock.patch("graphormer.pretrain.load_state_dict_from_url") as download:
            loaded = load_pretrained_model(
                pretrained_model_path=str(checkpoint_path)
            )

        download.assert_not_called()
        torch.testing.assert_close(
            loaded["encoder.weight"],
            self.model_state["encoder.weight"],
        )


if __name__ == "__main__":
    unittest.main()
