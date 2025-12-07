import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from OwnClassifier.LeafDataset import Classes
from pathlib import Path

EPOCHS = 20


def setup(classifierDir):
    statsFile = os.path.join(classifierDir, "stats", "stats.csv")

    stats_header = ["epoch", "trainLoss", "valLoss", "timeElapsed"]

    df = pd.read_csv(statsFile, header=None, names=stats_header)
    appendMeanAccuracy(Path(classifierDir), df)
    max_acc_idx = df["meanAccuracy"].idxmax()

    print(f"max accuracy is in epoch {max_acc_idx}")

    # plot loss
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.plot(df["epoch"], df["trainLoss"], label="Train loss")
    ax1.plot(df["epoch"], df["valLoss"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss vs Validation Loss")
    ax1.legend()
    ax2 = ax1.twinx()
    ax1.legend()
    plt.tight_layout()
    plt.show()


    return df, max_acc_idx


def appendMeanAccuracy(classifierDir, df):
    confusion = np.zeros((len(Classes), len(Classes)), dtype=int)
    meanAccuracy = []
    for e in range(EPOCHS):
        diagonals = 0
        nonDiagonals = 0
        epochOutputDir = os.path.join(classifierDir, f"epoch{e:02d}")
        for i, predC in enumerate(list(Classes)):
            for j, trueC in enumerate(list(Classes)):
                predToGtDir = f"P_{predC}_T_{trueC}"
                path = os.path.join(epochOutputDir, predToGtDir)

                files = len([f for f in os.listdir(path) if not f.startswith('.')])

                confusion[i, j] = files

                if i == j:
                    diagonals += files
                else:
                    nonDiagonals += files
        successRate = diagonals / (nonDiagonals + diagonals)

        meanAccuracy.append(successRate)

    df["meanAccuracy"] = meanAccuracy
    return df


def displayConfusionMatrix(classifierDir, e, title):
    confusion = np.zeros((len(Classes), len(Classes)), dtype=int)
    xlabels = [f"Pred {clas.value}" for clas in Classes]
    ylabels = [f"Actual {clas.value}" for clas in Classes]

    epochOutputDir = os.path.join(classifierDir, f"epoch{e:02d}")
    for i, predC in enumerate(list(Classes)):
        for j, trueC in enumerate(list(Classes)):
            predToGtDir = f"P_{predC}_T_{trueC}"
            path = os.path.join(epochOutputDir, predToGtDir)

            files = [f for f in os.listdir(path) if not f.startswith('.')]

            confusion[j, i] = len(files)

    plt.figure(figsize=(10, 8))

    cmap = sns.color_palette("flare", as_cmap=True)
    cmap.set_under("gray")
    print("ℹ️ for labels see LeadDataset.py and its Claasses enum")
    sns.heatmap(confusion,
                annot=confusion,
                vmin=1,
                fmt='',
                cmap=cmap,
                xticklabels=xlabels,
                yticklabels=ylabels,
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='black')

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return confusion

def worstClassesToPredict(confusionM, N=5):
    class_accuracy = confusionM.diagonal() / confusionM.sum(axis=1)
    classToAccuracy = [[Classes(i + 1), f"accuracy: {class_accuracy}"] for i, class_accuracy in
                       enumerate(class_accuracy)]

    worst_indices = np.argsort(class_accuracy)
    worst_classes = [classToAccuracy[i] for i in worst_indices]
    print("worst 5 classes are: ")
    for wc in worst_classes[:N]:
        print(wc)
    return worst_classes[:N]

def worstsClassesErrors(worst_classes, confusionM):
    gtClass: int

    for gtClass in worst_classes[:5]:
        print("")
        gtClassName = gtClass[0].name
        worstClassIdx = gtClass[0].value - 1
        confusionRow = confusionM[worstClassIdx]
        worst_indices = list(reversed(np.argsort(confusionRow)))[:6]
        print(f"GT: {gtClassName} but model predicated:")
        for worstConf in worst_indices:
            if worstConf == worstClassIdx: continue
            if confusionRow[worstConf] == 0: continue
            print(
                f"    - {Classes(worstConf + 1).name} in {confusionRow[worstConf] / sum(confusionRow) * 100:.{1}f}% cases")

#%%
