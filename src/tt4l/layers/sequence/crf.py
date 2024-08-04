# @Time     : 2024/7/19 18:37
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @Reference: https://github.com/yumoh/torchcrf/blob/master/torchcrf/__init__.py
# @About    :

from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        """
        Conditional random field.

        This module implements a conditional random field [LMP01]_. The forward computation
        of this class computes the log likelihood of the given sequence of tags and
        emission score tensor. This class also has `~CRF.decode` method which finds
        the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

        Args:
            num_tags (int): Number of tags.
            batch_first (bool): Whether the first dimension corresponds to the size of a minibatch.

        """
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self._num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(self._num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self._num_tags))
        self.transitions = nn.Parameter(torch.empty(self._num_tags, self._num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.

        Returns:

        """
        # nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        # nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        # nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)
        nn.init.zeros_(self.transitions)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self._num_tags})'

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                reduction: str = 'sum') -> torch.Tensor:
        """
        Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (torch.Tensor):
                Emission score tensor of size `(seq_length, batch_size, num_tags)`
                if `batch_first` is `False`, `(batch_size, seq_length, num_tags)` otherwise.
            tags (torch.Tensor):
                Sequence of tags tensor of size `(seq_length, batch_size)`
                if `batch_first` is `False`,
                `(batch_size, seq_length)` otherwise.
            attention_mask (torch.Tensor default None):
                Mask tensor of size `(seq_length, batch_size)`
                if `batch_first` is `False`, `(batch_size, seq_length)` otherwise.
            reduction (str default 'sum'):
                Specifies  the reduction to apply to the output:
                `none|sum|mean|token_mean`.
                `none`: no reduction will be applied.
                `sum`: the output will be summed over batches.
                `mean`: the output will be averaged over batches.
                `token_mean`: the output will be averaged over tokens.

        Returns:
            torch.Tensor:
                The log likelihood. This will have size `(batch_size,)`

        """
        self._validate(emissions, tags=tags, attention_mask=attention_mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if attention_mask is None:
            attention_mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            attention_mask = attention_mask.transpose(0, 1)
        # shape: (batch_size,)
        numerator = self._compute_score(emissions=emissions, tags=tags, attention_mask=attention_mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, attention_mask)
        # shape: (batch_size,)
        llh = numerator - denominator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        if reduction == 'token_mean':
            return llh.sum() / attention_mask.sum()

    @torch.jit.export
    def decode(self, emissions: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
               padding: bool = False) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (torch.Tensor):
                Emission score tensor of size
                `(seq_length, batch_size, num_tags)` if `batch_first` is `False`,
                `(batch_size, seq_length, num_tags)` otherwise.
            attention_mask (torch.Tensor):
                Mask tensor of size `(seq_length, batch_size)`
                if `batch_first` is `False`, `(batch_size, seq_length)` otherwise.
            padding (bool default `False`):
                Padding to max length

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, attention_mask=attention_mask)
        if attention_mask is None:
            attention_mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            attention_mask = attention_mask.transpose(0, 1)

        return self._viterbi_decode(emissions, attention_mask, padding=padding)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self._num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self._num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {(emissions.shape[0], emissions.shape[1])} and {(tags.shape[0], tags.shape[1])}'
                )

        if attention_mask is not None:
            if emissions.shape[:2] != attention_mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and attention_mask must match, '
                    f'got {(emissions.shape[0], emissions.shape[1])} and '
                    f'{(attention_mask.shape[0], attention_mask.shape[1])}'
                )
            no_empty_seq = not self.batch_first and attention_mask[0].all()
            no_empty_seq_bf = self.batch_first and attention_mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('attention_mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            emissions (torch.Tensor): `(seq_length, batch_size, num_tags)`
            tags (torch.Tensor): (seq_length, batch_size)
            attention_mask (tags: torch.Tensor): (seq_length, batch_size)

        Returns:

        """
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self._num_tags
        assert attention_mask.shape == tags.shape
        assert attention_mask[0].all()

        seq_length, batch_size = tags.shape
        attention_mask = attention_mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (attention_mask == 1)
            # shape: (batch_size,)
            tag = torch.where(attention_mask[i] > 0, tags[i], 0)  # 填充任何小于 num_tags 的值都可以
            last_tag = torch.where(attention_mask[i - 1] > 0, tags[i - 1], 0)
            _sel_length = attention_mask[i].sum().detach().cpu().item()
            score += self.transitions[last_tag, tag] * attention_mask[i]

            # Emission score for next tag, only added if next timestep is valid (attention_mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tag] * attention_mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = attention_mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            emissions (torch.Tensor): `(seq_length, batch_size, num_tags)`
            attention_mask (tags: torch.Tensor): (seq_length, batch_size)

        Returns:

        """
        assert emissions.dim() == 3 and attention_mask.dim() == 2
        assert emissions.shape[:2] == attention_mask.shape
        assert emissions.size(2) == self._num_tags
        assert attention_mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (attention_mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(attention_mask[i].unsqueeze(1) > 0, next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, attention_mask: torch.Tensor,
                        padding: bool) -> List[List[int]]:
        """

        Args:
            emissions (torch.Tensor): `(seq_length, batch_size, num_tags)`
            attention_mask (tags: torch.Tensor): (seq_length, batch_size)
            padding (bool):

        Returns:

        """
        assert emissions.dim() == 3 and attention_mask.dim() == 2
        assert emissions.shape[:2] == attention_mask.shape
        assert emissions.size(2) == self._num_tags
        assert attention_mask[0].all()

        seq_length, batch_size = attention_mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history: List[torch.Tensor] = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (attention_mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(attention_mask[i].unsqueeze(1) > 0, next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = attention_mask.long().sum(dim=0) - 1
        best_tags_list: List[List[int]] = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags: List[int] = []
            best_tags.append(best_last_tag.item())

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            # NOTE: reversed() cannot be used here because it is not supported by TorchScript,
            # see https://github.com/pytorch/pytorch/issues/31772.
            for hist in history[:seq_ends[idx]][::-1]:
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            # Padding
            while (len(best_tags) < seq_length) and padding:
                best_tags.append(-100)

            best_tags_list.append(best_tags)

        return best_tags_list
