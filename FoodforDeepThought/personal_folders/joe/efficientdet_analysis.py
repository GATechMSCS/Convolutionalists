from unittest.mock import right

import pandas as pd
import matplotlib.pyplot as plt


def run():
    df = pd.read_csv('summaries/summary3.csv', index_col=0)
    df.rename(columns={"train_loss": "Training Loss", "eval_loss": "Validation Loss", "eval_map": "Validation mAP"}, inplace=True)
    loss_df = df.iloc[:, :2]
    map_df = df.iloc[:, 2:]
    print(loss_df.head())
    fig, ax1 = plt.subplots()
    loss_df.plot(ax=ax1)
    ax1.set_title('EfficientDet Learning Curve')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    map_df.plot(ax=ax2, color='red')
    ax2.set_ylabel("mAP")
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(True)
    ax2.grid(linestyle='--')
    ax2.set_xlabel("Epoch")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='center right', facecolor='white', framealpha=1)
    ax1.legend().remove()
    # ax2.set_axisbelow(True)
    plt.savefig(fname='efficientdet-learning-curve.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    run()
