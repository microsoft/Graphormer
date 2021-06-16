# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['c', 's', 'm', 'l']

    def forward(self, xs):
        assert isinstance(xs, list) or isinstance(xs, tuple)

        if self.mode == 'c':
            return torch.cat(xs, dim=-1)
        elif self.mode == 's':
            nr = 0
            for x in xs:
                nr += x
            return nr
        elif self.mode == 'm':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'l':
            return xs[-1]

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
