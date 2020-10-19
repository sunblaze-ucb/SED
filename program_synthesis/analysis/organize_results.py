import pandas as pd

def by_max_step(accuracies):
    max_step = accuracies.groupby(list(set(accuracies) - {'Step', 'Accuracy', 'Exact'})).transform(max).Step
    return accuracies[accuracies.Step == max_step]

def by_max_acc(accuracies):
    max_acc = accuracies.groupby(list(set(accuracies) - {'Step', 'Accuracy', 'Exact'})).transform(max).Accuracy
    possible_dup_max_acc = accuracies[accuracies.Accuracy == max_acc]
    # deduplicate
    return by_max_step(possible_dup_max_acc)

def _add_processed_row(df_orig, df, model, data_source, s, k):    
    if s == "X":
        strategy = data_source
    else:
        start_with_beams = data_source != "nearai"
        s_actually_used = "best_first" if k == 1 else {"G" : "greedy", "B" : "best_first"}[s]
        descriptor = [data_source, s_actually_used, str(k)] + ["start-with-beams"] * start_with_beams
        strategy = ",,".join(descriptor)


    filtered = df_orig[(df_orig.DataLabel == strategy) & (df_orig.Model == model)]
    if len(filtered) == 0:
        print(model, strategy)
        return
    assert len(filtered) == 1, str(filtered)
    row = filtered.iloc[0]
    df.append([model, data_source, s, k, 100 * (1 - row.Accuracy), 100 * (1 - row.Exact)])

K_VALS = 1, 5, 10, 25, 50, 100, 200

def separate_strategies(df_orig):
    df_orig = by_max_acc(df_orig)
    df = []
    for data_source in "nearai", "nearai32", "egnps64", "egnpsgood":
        for model in set(df_orig.Model):
            for s in "G", "B":
                for k in K_VALS:
                    _add_processed_row(df_orig, df, model, data_source, s, k)
            _add_processed_row(df_orig, df, model, data_source, "X", 1)
    return pd.DataFrame(df, columns=["model", "synthesizer", "strategy", "debugger steps", "gen error", "exact error"])

def by_k(topline):
    columns_except_k = [col for col in topline if col != 'debugger steps' and col.split()[-1] != 'error']
    result = {}
    def set_value(key, k, row):
        if key not in result:
            result[key] = list(key) + [''] * len(K_VALS) * 2
        loc = len(key) + K_VALS.index(k)
        assert result[key][loc] == ''
        assert result[key][loc + len(K_VALS)] == ''
        result[key][loc] = row['gen error']
        result[key][loc + len(K_VALS)] = row['exact error']

    for _, row in topline.iterrows():
        k = row['debugger steps']
        set_value(tuple(row[c] for c in columns_except_k), k, row)
    
    columns = columns_except_k + ["gen error %s" % k for k in K_VALS] + ["exact error %s" % k for k in K_VALS]
    return pd.DataFrame(list(result.values()), columns=columns)

def to_display_name(model):
    return {"nearai" : "LGRL-GD", "nearai32" : "LGRL", "egnpsgood" : "EGNPS"}[model]