import sys
import os
import time
import collections

import torch
from torch.autograd import Variable
import torch.nn.functional as F


# TODO: Make an argument (dataset arg?).
MAX_DECODER_LENGTH = 100


def expand_by_beam(v, beam_size):
    # v: batch size x ...
    # output: (batch size * beam size) x ...
    return (v.unsqueeze(1).repeat(1, beam_size, *([1] * (v.dim() - 1))).
            view(-1, *v.shape[1:]))


class BeamSearchMemory(object):
    '''Batched memory used in beam search.'''
    __slots__ = ()

    def expand_by_beam(self, beam_size):
        '''Return a copy of self where each item has been replicated
        `beam_search` times.'''
        raise NotImplementedError


class BeamSearchState(object):
    '''Batched recurrent state used in beam search.'''
    __slots__ = ()

    def select_for_beams(self, batch_size, indices):
        '''Return the hidden state necessary to continue the beams.

        batch_size: int
        indices: 2 x batch size * beam size LongTensor
        '''
        raise NotImplementedError


BeamSearchResult = collections.namedtuple('BeamSearchResult', ['sequence', 'total_log_prob', 'log_probs', 'log_probs_torch'])

def length_penalty(sequence_lengths, penalty_factor):
    """ 
    Calculate the length penalty according to 
    https://arxiv.org/abs/1609.08144
    lp(Y) =(5 +|Y|)^α / (5 + 1)^α

    Input:
    sequence_lenthgs: the sequences length of all hypotheses of size [batch size x beam size x vocab size]
    penalty_factor: A scalar that weights the length penalty.
    
    Returns:
    The length penalty factor, a tensor fo shape [batch size x beam size].
    """
    return torch.div((5. + sequence_lengths)**penalty_factor, (5. + 1.)
                **penalty_factor)

def hyp_score(log_probs, sequence_lengths, penalty_factor):
    """
    Calculates scores for beam search hypotheses.
    """

    # Calculate the length penality
    length_penality_ = length_penalty(
        sequence_lengths=sequence_lengths,
        penalty_factor=penalty_factor)

    score = log_probs / length_penality_
    
    return score, length_penality_

def Calculate_prob_idx_token_with_penalty(cuda, step, sizes, total_log_probs, finished_penalty, lengths_penalty, factor):

    """
    Calculate prev_probs using the length penalty

    Returns:
    the previous probabilities, their indices and the previously predicted tokens
    """
    tt = torch.cuda if cuda else torch

    batch_size,actual_beam_size,logit_size = sizes

    if step>0:

        # Clone log probs
        curr_scores = total_log_probs.clone()

        lengths_to_add = Variable(tt.FloatTensor(batch_size*actual_beam_size*logit_size).fill_(0)).view(batch_size,actual_beam_size,logit_size)
        add_mask = (1 - finished_penalty)
        lengths_to_add = add_mask.repeat(1, logit_size).view(batch_size,actual_beam_size,logit_size) * lengths_to_add

        new_prediction_lengths = lengths_to_add + lengths_penalty.repeat(1, logit_size).view(batch_size,actual_beam_size,logit_size).float()

        # Compute length penalty
        curr_scores, lp = hyp_score(
        log_probs=curr_scores,
        sequence_lengths=new_prediction_lengths,
        penalty_factor=factor)
        curr_scores = curr_scores.view(batch_size, -1)

        # Recover log probs
        prev_probs, indices = curr_scores.topk(actual_beam_size, dim=1)

        prev_tokens = (indices % logit_size)

        target_len = []
        for b in range(batch_size):
            for be in range(actual_beam_size):
                target_len.append(lp[b][be][prev_tokens[b][be]])

        target_len = torch.stack(target_len).view(batch_size, actual_beam_size)
        prev_probs = torch.mul(prev_probs, target_len) 

        # Calculate the length and mask of the next predictions.
        # 1. Finished beams remain unchanged
        # 2. Beams that are now finished (EOS predicted) remain unchanged
        # 3. Beams that are not yet finished have their length increased by 1
        eos = prev_tokens == 1  

        finished_penalty = (finished_penalty==1) | eos
        finished_penalty = finished_penalty.float().view(batch_size, actual_beam_size)

        lengths_to_add = (1 - finished_penalty)
        next_prediction_len = target_len
        next_prediction_len += lengths_to_add

        lengths_penalty = next_prediction_len
        prev_tokens = prev_tokens.view(-1)
    
        return prev_probs, indices, prev_tokens, finished_penalty, lengths_penalty
    
    else:
        log_probs_flat = total_log_probs.view(batch_size, -1)
        prev_probs, indices = log_probs_flat.topk(actual_beam_size, dim=1)
        prev_tokens = (indices % logit_size).view(-1)

        return prev_probs, indices, prev_tokens, finished_penalty, lengths_penalty



def beam_search(*args, volatile=True, **kwargs):
    if volatile:
        with torch.no_grad():
            return beam_search_(*args, **kwargs)
    else:
        return beam_search_(*args, **kwargs)

def beam_search_(batch_size,
                enc,
                masked_memory,
                decode_fn,
                beam_size,
                cuda=False,
                max_decoder_length=MAX_DECODER_LENGTH,
                return_attention=False,
                return_beam_search_result=False,
                differentiable=False,
                use_length_penalty=False,
                factor = 0.7):
    # enc: batch size x hidden size
    # memory: batch size x sequence length x hidden size
    tt = torch.cuda if cuda else torch
    prev_tokens = Variable(tt.LongTensor(
        batch_size).fill_(0))
    prev_probs = Variable(tt.FloatTensor(
        batch_size, 1).fill_(0))
    prev_hidden = enc
    finished = [[] for _ in range(batch_size)]
    result = [[BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0, log_probs_torch=[])
               for _ in range(beam_size)] for _ in range(batch_size)]
    batch_finished = [False for _ in range(batch_size)]
    # b_idx: 0, ..., 0, 1, ..., 1, ..., b, ..., b
    # where b is the batch size, and each group of numbers has as many elements
    # as the beam size.
    b_idx = Variable(
        torch.arange(0, batch_size, out=torch.LongTensor()).unsqueeze(1).repeat(1, beam_size).view(-1))

    finished_penalty=Variable(tt.FloatTensor(batch_size*beam_size).fill_(0)).view(batch_size, beam_size)
    lengths_penalty=Variable(tt.LongTensor(batch_size*beam_size).fill_(0)).view(batch_size, beam_size)

    prev_masked_memory = masked_memory.expand_by_beam(beam_size)
    attn_list = [] if return_attention else None
    for step in range(max_decoder_length):
        hidden, logits = decode_fn(prev_tokens, prev_hidden,
                                prev_masked_memory if step > 0 else
                                masked_memory, attentions=attn_list)

        logit_size = logits.size(1)
        # log_probs: batch size x beam size x vocab size
        log_probs = F.log_softmax(logits, dim=-1).view(batch_size, -1, logit_size)
        total_log_probs = log_probs + prev_probs.unsqueeze(2)
        # log_probs_flat: batch size x beam_size * vocab_size
        log_probs_flat = total_log_probs.view(batch_size, -1)
        # indices: batch size x beam size
        # Each entry is in [0, beam_size * vocab_size)
        actual_beam_size = min(beam_size, log_probs_flat.size(1))
        sizes=(batch_size,actual_beam_size,logit_size)

        if use_length_penalty:
            prev_probs, indices, prev_tokens, finished_penalty, lengths_penalty = Calculate_prob_idx_token_with_penalty(cuda, step, sizes, total_log_probs, finished_penalty, lengths_penalty, factor)

        else:

            prev_probs, indices = log_probs_flat.topk(actual_beam_size, dim=1)
            if beam_size != actual_beam_size:
                prev_probs = torch.cat([prev_probs, -10000 + prev_probs[:, 0].unsqueeze(1).repeat(1, beam_size - actual_beam_size)], dim=1)
                indices = torch.cat([indices, indices[:, 0].unsqueeze(1).repeat(1, beam_size - actual_beam_size)], dim=1)
        
            # prev_tokens: batch_size * beam size
            # Each entry indicates which token should be added to each beam.
            prev_tokens = (indices % logit_size).view(-1)

        # This takes a lot of time... about 50% of the whole thing.
        indices = indices.cpu()
        # k_idx: batch size x beam size
        # Each entry is in [0, beam_size), indicating which beam to extend.
        k_idx = (indices / logit_size)

        b_idx_to_use = b_idx

        idx = torch.stack([b_idx_to_use, k_idx.view(-1)])
        # prev_hidden: (batch size * beam size) x hidden size
        # Contains the hidden states which produced the top-k (k = beam size)
        # tokens, and should be extended in the step.
        prev_hidden = hidden.select_for_beams(batch_size, idx)

        prev_result = result
        result = [[] for _ in range(batch_size)]
        can_stop = True
        prev_probs_np = prev_probs.data.cpu().numpy()
        log_probs_np = log_probs.data.cpu().numpy()
        k_idx = k_idx.data.numpy()
        indices = indices.data.numpy()
        for batch_id in range(batch_size):
            if batch_finished[batch_id]:
                continue
            # print(step, finished[batch_id])
            if len(finished[batch_id]) >= beam_size:
                # If last in finished has bigger log prob then best in topk, stop.
                if finished[batch_id][-1].total_log_prob > prev_probs_np[batch_id, 0]:
                    batch_finished[batch_id] = True
                    continue
            for idx in range(beam_size):
                token = indices[batch_id, idx] % logit_size
                kidx = k_idx[batch_id, idx]
                # print(step, batch_id, idx, 'token', token, kidx, 'prev', prev_result[batch_id][kidx], prev_probs.data[batch_id][idx])
                if token == 1:  # 1 == </S>
                    if differentiable:
                        log_probs_torch = prev_result[batch_id][kidx].log_probs_torch + [log_probs[batch_id, kidx, token]]
                    else:
                        log_probs_torch = None
                    finished[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence,
                        total_log_prob=prev_probs_np[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs_np[batch_id, kidx, token]],
                        log_probs_torch=log_probs_torch))
                    result[batch_id].append(BeamSearchResult(sequence=[], log_probs=[], total_log_prob=0, log_probs_torch=[]))
                    prev_probs.data[batch_id][idx] = float('-inf')
                else:
                    if differentiable:
                        log_probs_torch = prev_result[batch_id][kidx].log_probs_torch + [log_probs[batch_id, kidx, token]]
                    else:
                        log_probs_torch = None
                    result[batch_id].append(BeamSearchResult(
                        sequence=prev_result[batch_id][kidx].sequence + [token],
                        total_log_prob=prev_probs_np[batch_id, idx],
                        log_probs=prev_result[batch_id][kidx].log_probs + [log_probs_np[batch_id, kidx, token]],
                        log_probs_torch=log_probs_torch))
                    can_stop = False
            if len(finished[batch_id]) >= beam_size:
                # Sort and clip.
                finished[batch_id] = sorted(
                    finished[batch_id], key=lambda x: -x.total_log_prob)[:beam_size]
        if can_stop:
            break

    for batch_id in range(batch_size):
        # If there is deficit in finished, fill it in with highest probable results.
        if len(finished[batch_id]) < beam_size:
            i = 0
            while i < beam_size and len(finished[batch_id]) < beam_size:
                if result[batch_id][i]:
                    finished[batch_id].append(result[batch_id][i])
                i += 1

    if not return_beam_search_result:
        for batch_id in range(batch_size):
            finished[batch_id] = [x.sequence for x in finished[batch_id]]

    if return_attention:
        # all elements of attn_list: (batch size * beam size) x input length
        attn_list[0] = expand_by_beam(attn_list[0], beam_size)
        # attns: batch size x bean size x out length x inp length
        attns = torch.stack(
                [attn.view(batch_size, -1, attn.size(1)) for attn in attn_list],
                dim=2)
        return finished, attns
    return finished
