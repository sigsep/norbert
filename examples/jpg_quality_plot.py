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
    df['size'] = df['size'].astype(int)
    df['bitrate'] = df['size'] / 7.0
    df['quality'] = df['quality'].astype(int)
    df['ODG'] = df['ODG'].astype(int)
    agg = df.groupby(['quality'], as_index=False).mean()
    min_quality = min(agg[agg['ODG'] > -1]['quality'])

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = agg.plot(x="quality", y="ODG", kind="line", ax=ax, color='b', legend=False)
    ax2 = ax.twinx()
    agg.plot(x="quality", y="bitrate", kind="line", ax=ax2, color="r", legend=False)
    ax.set_ylabel("PEAQ ODG Score", color='b')
    ax2.set_ylabel("Bitrate in bps", color='r')
    ax.set_xlabel("JPEG Quality Parameter")
    ax.axvspan(min_quality, 100, facecolor='#2ca02c', alpha=0.2)
    ax.tick_params('y', colors='b')
    ax2.tick_params('y', colors='r')
    plt.title("Reconstruction Error (Mono)")
    plt.savefig("mono.png")
    plt.show()
