# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
import numpy as np
import tqdm
from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed.lib.query import Q


def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)


def print_table(table, header_text, row_labels, col_labels, colwidth=10, latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = [
            "\\textbf{" + str(col_label).replace("%", "\\%") + "}" for col_label in col_labels
        ]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")


def print_results_tables(records):
    """Given all records, print a results table for each dataset."""
    grouped_records = reporting.get_grouped_records(records).map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = (
        [n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS]
    )

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_accs = (
                    grouped_records.filter_equals(
                        "dataset, algorithm, test_env", (dataset, algorithm, test_env)
                    ).select("sweep_acc")
                )
                # import pdb; pdb.set_trace()

                mean, err, table[i][j] = format_mean(trial_accs, latex)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.3f}".format(sum(means) / len(means))

        col_labels = ["Algorithm", *datasets.get_dataset_class(dataset).ENVIRONMENTS, "Avg"]
        header_text = (f"Dataset: {dataset}, " f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels), colwidth=20, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (
                grouped_records.filter_equals("algorithm, dataset", (algorithm, dataset)).
                group("trial_seed").map(lambda trial_seed, group: group.select("sweep_acc").mean())
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25, latex=latex)

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    line_parsed = json.loads(line[:-1])
                records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

if __name__ == "__main__":
    # goal: select val_acc, test_acc and lambda
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    records = reporting.load_records(args.input_dir)

    print("Total records:", len(records))

    print_results_tables(records)

