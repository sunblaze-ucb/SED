from abc import ABC, abstractmethod

from collections import defaultdict
from itertools import count

import torch

from datasets.executor import evaluate_code
from datasets.karel import mutation

from models.base import InferenceResult

from datasets.karel.utils import chunked
import numpy as np

class IterativeSearch:
    def __init__(self, original_inference, init_strategy, executor, add_trace, batch_processor, start_with_beams,
                 time_limit, overfit_model):
        self.original_inference = original_inference
        self.init_strategy = init_strategy
        self.executor = executor
        self.add_trace = add_trace
        self.batch_processor = batch_processor
        self.overfit_model = overfit_model
        # whether to start with the beams from the original model
        self.start_with_beams = start_with_beams
        self.time_limit = time_limit

    def __call__(self, batch):
        original_batch = batch

        strategies = [self.init_strategy(item) for item in batch.orig_examples]
        done = [False] * len(batch.orig_examples)
        finalized_candidates = [[] for _ in range(len(batch.orig_examples))]
        attempts = [[] for _ in range(len(batch.orig_examples))]
        index_mapping = {i: i for i in
                         range(len(batch.orig_examples))}  # mapping from indices in batch/strategies to indices in done
        num_inferences = 0
        for iteration_idx in count():
            if iteration_idx == 0 and self.start_with_beams:
                results = [example.ref_beams for example in batch.orig_examples]
                assert not any(
                    x is None for x in results), "the original examples must contain a full list of reference beams"
            else:
                results = [result.info['candidates'] for result in self.original_inference(batch)]
                num_inferences += 1
            decisions = [strategy.decide(result, lambda code: self.test_results(code, example)) for
                         strategy, result, example in zip(strategies, results, batch.orig_examples)]
            new_index_mapping = {}
            new_strategies = []
            new_wrong_code = []
            new_batch = []
            for idx, decision in enumerate(decisions):
                give_up = (num_inferences == self.time_limit or decision[0] == 'give_up') and not finalized_candidates[index_mapping[idx]]
                if decision[0] == 'accept' or give_up:
                    finalized_candidates[index_mapping[idx]].append(decision[1])

                if decision[0] == 'accept' and self.overfit_model is None or decision[0] == 'give_up':
                    # only mark something done if the overfit model is not in effect,
                    # otherwise we keep going until we run out of time
                    # except in the case where we give up, in which case we stop immediately to avoid syntax errors
                    done[index_mapping[idx]] = True
                else:
                    attempts[index_mapping[idx]].append(decision[1])
                    new_wrong_code.append(decision[1])
                    new_batch.append(batch.orig_examples[idx])
                    new_strategies.append(strategies[idx])
                    new_index_mapping[len(new_wrong_code) - 1] = index_mapping[idx]

            if all(done) or num_inferences == self.time_limit:
                break
            index_mapping = new_index_mapping
            strategies = new_strategies
            batch = self.update_wrong_code_and_pack(new_batch, new_wrong_code)

        if self.overfit_model is None:
            assert all(len(candidates) == 1 for candidates in finalized_candidates)
            best_code = [candidates[0] for candidates in finalized_candidates]
        else:
            best_code = self.select_best_code_per(original_batch, finalized_candidates)

        return [InferenceResult(code_sequence=code, info={'candidates': [code], 'expanded': [expanded]})
                for code, expanded in zip(best_code, attempts)]

    def update_wrong_code_and_pack(self, new_examples, new_wrong_code, add_trace=None, batch_processor=None):
        if add_trace is None:
            # this case is when we are using the forward model not the overfit model which may
            # need the trace
            add_trace = self.add_trace
        if batch_processor is None:
            batch_processor = self.batch_processor
        assert new_examples
        assert len(new_examples) == len(new_wrong_code)
        updated_examples = []
        for example, code in zip(new_examples, new_wrong_code):
            updated_examples.append(
                mutation.add_incorrect_code(example, tuple(code), add_trace, self.executor,
                                            check_ref_example=False))
        return batch_processor(updated_examples)

    def test_results(self, code, example):
        return evaluate_code(code, example.schema.args, example.input_tests, self.executor.execute)

    def run_overfit_model(self, items):
        egs, cands = zip(*items)
        batch = self.update_wrong_code_and_pack(egs, cands, add_trace=True,
                                                batch_processor=self.overfit_model.batch_processor(for_eval=True))
        with torch.no_grad():
            return self.overfit_model.inference(batch)

    def select_best_code_per(self, original_batch, finalized_candidates):
        flattened_candidates = []
        flattened_examples = []
        indices_per_original = []
        for example, candidates in zip(original_batch.orig_examples, finalized_candidates):
            indices_per_original.append([])
            if len(candidates) == 1:
                continue
            for candidate in candidates:
                indices_per_original[-1].append(len(flattened_examples))
                flattened_examples.append(example)
                flattened_candidates.append(candidate)

        results = []
        for items in chunked(zip(flattened_examples, flattened_candidates), len(original_batch.orig_examples)):
            results += self.run_overfit_model(items).cpu().detach().numpy().tolist()
        best_code = []
        for idxs, candidates in zip(indices_per_original, finalized_candidates):
            if len(candidates) == 1:
                best_code.append(candidates[0])
                continue
            best_idx = max(range(len(candidates)), key=lambda i: results[idxs[i]])
            best_code.append(candidates[best_idx])
        return best_code


class Strategy(ABC):
    @abstractmethod
    def decide(self, candidates, evaluate):
        pass

    @staticmethod
    def get(descriptor):
        if ":" not in descriptor:
            descriptor += ":"
        start, *rest = descriptor.split(":")
        kwargs = eval("dict({})".format(":".join(*rest)))
        return {
            'greedy': lambda: GreedyStrategy,
            'best_first': lambda: BestFirstSearch,
            'diverse': lambda: DiversitySearch
        }[start](**kwargs)


def valid(considered_program, result):
    if not considered_program:
        return False
    if result['syntax-error'] > 0:
        return False
    return True


class GreedyStrategy(Strategy):
    def __init__(self, item):
        self.seen = set()
        del item  # no need

    def decide(self, candidates, evaluate):
        unseen = []
        for considered in candidates:
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if not valid(considered, res):
                continue
            if res['correct'] == res['total']:
                self.seen.add(considered)
                return 'accept', considered
            unseen.append((res['correct'], considered))
        if not unseen:
            self.seen.add(tuple(candidates[0]))
            return 'give_up', candidates[0]
        unseen.sort(reverse=True)
        self.seen.add(unseen[0][1])
        return 'expand', unseen[0][1]


class BestFirstSearch(Strategy):
    def __init__(self, item):
        self.seen = set()
        self.by_number_correct = defaultdict(list)

    def decide(self, candidates, evaluate):
        for considered in candidates:
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if not valid(considered, res):
                continue
            self.seen.add(considered)
            assert res['total'] == 5
            self.by_number_correct[res['correct']].append(considered)

        for n_correct in sorted(self.by_number_correct, reverse=True):
            if self.by_number_correct[n_correct]:
                decision = 'accept' if n_correct == 5 else 'expand'
                return decision, self.by_number_correct[n_correct].pop(0)

        return 'give_up', tuple(candidates[0])


class DiversitySearch(Strategy):
    """
    Add a new alternative to BestFirstStrategy that takes into
    account semantic diversity, that is it breaks ties by 
    picking programs that pass different sets of test cases 
    from ones expanded in the past. see res[‘individual’] 

    """
    def __init__(self, item):
        self.seen = set()
        self.seen_patterns = defaultdict(int)
        self.by_number_correct = defaultdict(list)

    def decide(self, candidates, evaluate):
        for considered in candidates:
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if not valid(considered, res):
                continue
            self.seen.add(considered)
            self.seen_patterns[tuple(res['individual'])] += 1
            assert res['total'] == 5
            self.by_number_correct[res['correct']].append((considered, res['individual']))

        for n_correct in sorted(self.by_number_correct, reverse=True):
            if self.by_number_correct[n_correct]:
                decision = 'accept' if n_correct == 5 else 'expand'
                return decision, self.diverse_decision(self.by_number_correct[n_correct])

        return 'give_up', tuple(candidates[0])

    def diverse_decision(self, items):
        to_use_idx = min(range(len(items)), key=lambda i: self.seen_patterns[tuple(items[i][1])])
        to_use, _ = items.pop(to_use_idx)
        return to_use
