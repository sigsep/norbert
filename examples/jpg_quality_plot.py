import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pickle',
    )
    args = parser.parse_args()

    df = pd.read_pickle(args.pickle)
    print(df)
    g = sns.factorplot(
        x="quality",
        y="ODG",
        data=df,
        legend_out=False,
        dodge=False,
    )

    plt.legend(loc='upper left', title="JPG Quality (Mono)")
    plt.savefig("mono.png")
