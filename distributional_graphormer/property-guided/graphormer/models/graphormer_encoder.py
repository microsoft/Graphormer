from builtins import hasattr
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import FairseqEncoder

from fairseq.modules import (
    LayerNorm,
)
from ..modules import GraphormerGraphEncoder

logger = logging.getLogger(__name__)


class GraphormerEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = args.max_nodes

        self.graph_encoder = self.build_graph_encoder(args)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

    def build_graph_encoder(self, args):
        return GraphormerGraphEncoder(args)

    def make_padding_mask(self, batched_data):
        encoder_padding_mask = (batched_data["x"][:, :, 0]).eq(0)  # B x T
        # prepend 1 for CLS token
        B_zeros = torch.zeros(
            (encoder_padding_mask.size(0), 1),
            dtype=torch.bool,
            device=encoder_padding_mask.device,
        )
        encoder_padding_mask = torch.cat(
            [B_zeros, encoder_padding_mask], dim=1
        ).contiguous()
        return encoder_padding_mask

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        x = x + self.lm_output_learned_bias

        encoder_padding_mask = self.make_padding_mask(batched_data)

        src_lengths = (
            (~encoder_padding_mask)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )

        return {
            "encoder_out": [x.transpose(0, 1)],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [x],  # B x T x C
            "encoder_states": inner_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def max_positions(self):
        return self.max_nodes

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }
