import torch
from torch.utils.data import Sampler


class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if (self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

