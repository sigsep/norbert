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


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = agg.plot(x="quality", y="ODG", kind="line", ax=ax, color=colors[0], legend=False)
    ax2 = ax.twinx()
    agg.plot(x="quality", y="bitrate", kind="line", ax=ax2, color=colors[1], legend=False)
    ax.set_ylabel("PEAQ ODG Score", color=colors[0])
    ax2.set_ylabel("Bitrate in bps", color=colors[1])
    ax.set_xlabel("JPEG Quality Parameter")
    ax.axhline(y=-1.0, linewidth=1, color='k', linestyle='dashed')
    ax.axvspan(min_quality, 100, facecolor='#2ca02c', alpha=0.2)
    ax.tick_params('y', colors=colors[0])
    ax2.tick_params('y', colors=colors[1])
    ax.text(
        min_quality + 1, -1.7, "Perceptible but not annoying",
        size=9
    )
    plt.title("Reconstruction Error (Stereo)")
    plt.savefig("mono.png")
    plt.show()
