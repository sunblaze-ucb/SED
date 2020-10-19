import glob
import os
import re
import json
from functools import lru_cache

import pandas as pd
from program_synthesis.datasets import dataset

NAME_REGEX = r"((checkpoint|events|args).*)|report-dev-m(?P<d>[^-]*(start-with-beams)?(,,overfit=[a-z_]*)?)-(?P<s>[0-9]*)-(?P<t>train|eval|real)\.jsonl$"

def get_exact_match(path):
    exact_match_path = path + ".exact_match.json"
    if not os.path.exists(exact_match_path):
        try:
            with open(path) as f:
                contents = [json.loads(line) for line in f]
        except json.JSONDecodeError:
            assert False, path
        num_exact_match = sum(res['example']['code'] == res['code']['code_sequence'] for res in contents[1:])
        result = dict(exact=num_exact_match, total=len(contents[1:]))
        with open(exact_match_path, "w") as f:
            json.dump(result, f)
    try:
        with open(exact_match_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(exact_match_path)
        raise

def table_of_accuracies(label, pbar=lambda x: x):
    print(label)
    if "327" in label:
        logdirs = ["../logdirs/" + label]
    else:
        logdirs =  glob.glob("../logdirs/{},*".format(label))
    if "327" not in label:
        logdirs = [l for l in logdirs if "327" not in l]
    if not logdirs:
        raise RuntimeError("nothing found {}".format(label))
    [logdir] = logdirs
    data = []
    for f in pbar(os.listdir(logdir)):
        if not f.endswith(".jsonl"): continue
        m = re.match(NAME_REGEX, f)
        assert m and m.group("d") is not None, "{} not a valid name".format(f)
        data_source = m.group("t")
        data_label = m.group("d")
        checkpoint = int(m.group("s"))
        datatype = m.group("t")
        path = os.path.join(logdir, f)
        with open(path) as fp:
            try:
                stats = json.loads(next(fp))
            except StopIteration:
                stats = None
        if stats is None or not stats.get('done', stats['total'] >= 2500):
            continue

        exact_match = get_exact_match(path)
        assert exact_match['total'] == stats['total'], str((exact_match, stats, path))

        data.append([label, checkpoint, stats['correct'] / stats['total'], stats['total'], exact_match['exact'] / stats['total'], data_source, data_label])
    df = pd.DataFrame(
        data, columns=['Model', 'Step', 'Accuracy', 'Total', 'Exact', 'DataSource', 'DataLabel']
    )
    return df

@lru_cache(None)
def get_baseline_stats(model=None, segment="val", path=None, data_folder='../data'):
    if path is None:
        assert model is not None
        path = "../baseline/{}-{}.json".format(model, segment)
    dset = dataset.KarelTorchDataset('{}/karel/{}.pkl'.format(data_folder, segment))
    with open(path) as f:
        results = json.load(f)
    exact = sum(x['output'] == y.code_sequence for x, y in zip(results, dset))
    correct = sum(x['is_correct'] for x in results)
    passes_given_tests = sum(x['passes_given_tests'] for x in results)
    return dict(exact=exact, correct=correct, passes_given=passes_given_tests)

def get_for_baseline_model(accuracies, baseline_model):
    baseline = get_baseline_stats(baseline_model, 'val' if baseline_model != "egnpsgood" else "test")

    acs = accuracies[accuracies.DataLabel.map(lambda x: x.split(",")[0] == baseline_model and ',,overfit=' not in x)].copy()
    acs.Accuracy *= acs.Total
    acs.Accuracy += baseline['correct']
    acs.Accuracy /= 2500
    acs.Exact *= acs.Total
    acs.Exact += baseline['exact']
    acs.Exact /= 2500
    acs = pd.concat([acs, accuracies[accuracies.DataLabel.map(lambda x: x.split(",")[0] == baseline_model and ',,overfit=' in x)].copy()])
    acs = acs[['Model', 'Step', 'DataLabel', 'Accuracy', 'Exact']].copy()
    return acs

def get_for_baseline_models(accuracies, *baseline_models):
    return pd.concat([get_for_baseline_model(accuracies, b) for b in baseline_models])
