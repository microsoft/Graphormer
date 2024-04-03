import numpy as np


class AAVocab:
    AMINO = "ACDEFGHIKLMNPQRSTVWY"
    assert len(AMINO) == 20
    unk_idx = 20
    mask_idx = 21
    pad_idx = 22
    _vocab = np.full(128, unk_idx, dtype=np.uint8)
    _vocab[np.array(AMINO, "c").view(np.uint8)] = np.arange(len(AMINO))
    _vocab.setflags(write=False)

    @staticmethod
    def encode(seq):
        assert isinstance(seq, str)
        return AAVocab._vocab[np.array(seq, "c").view(np.uint8)]

    @staticmethod
    def size():
        return 23
