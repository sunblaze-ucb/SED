
import os

import argparse
import json
import tqdm

import numpy as np

from program_synthesis.datasets import dataset, executor

parser = argparse.ArgumentParser()
parser.add_argument("--data-pickle", default='data/karel/val.pkl')
parser.add_argument("--input-file", required=True, help="file containing list of beams, each of which is a list of programs")
parser.add_argument("--output-file", required=True)

args = parser.parse_args()

assert not os.path.exists(args.output_file)

with open(args.input_file) as f:
    programs = json.load(f)

examples = dataset.KarelTorchDataset(
    args.data_pickle,
    lambda x: x)


def evaluate_code(eg, beam):
    exe = executor.KarelExecutor()
    tests = []
    tests += list(eg.input_tests)
    tests += list(eg.tests)

    stats = executor.evaluate_code(beam[0], eg.schema.args, tests, exe.execute)
    prediction = dict(
        output=beam[0],
        beams=beam,
        beams_correct=[executor.evaluate_code(hypothesis, eg.schema.args, tests, exe.execute) for hypothesis in beam],
        is_correct=stats['correct'] == stats['total'],
        individual=stats['individual'],
        passes_given_tests=all(stats['individual'][:len(eg.input_tests)])

    )
    return prediction

result = [evaluate_code(eg, beams) for eg, beams in zip(examples, tqdm.tqdm(programs))]

with open(args.output_file, "w") as f:
    json.dump(result, f)