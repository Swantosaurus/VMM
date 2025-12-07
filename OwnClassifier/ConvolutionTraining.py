import os
import torch
import time
import csv
import shutil
from torch.utils.data import DataLoader
from LeafDataset import Classes, ClassMapper, LeafDataset
from ConvolutionClassifierModel import ConvolutionClassifierModel

OUTPUT_DIR = "Convolution-01-256im"
MODELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
STATS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "stats")
STATS_FILE = os.path.join(STATS_OUTPUT_DIR, "stats.csv")
EPOCHS = 20
INPUT_DATASET_DIR = "../dataset"

TRAINING_FROM = 0
TRAINING_TO = 800

VALIDATE_FROM = 800
VALIDATE_TO = 1000

trainDatates = LeafDataset(INPUT_DATASET_DIR, TRAINING_FROM, TRAINING_TO)
validateDataset = LeafDataset(INPUT_DATASET_DIR, VALIDATE_FROM, VALIDATE_TO)

trainLoader = DataLoader(trainDatates, batch_size=32, shuffle=True)
validateLoader = DataLoader(validateDataset, batch_size=32)

os.makedirs(OUTPUT_DIR, exist_ok=False)
os.makedirs(STATS_OUTPUT_DIR, exist_ok=False)
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=False)

print(f"trainLoader size {len(trainLoader)}")
print(f"validateLoader size {len(validateLoader)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available else device

print(f"device used: {device}")

model = ConvolutionClassifierModel(29).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lossFn = torch.nn.CrossEntropyLoss()

startTime = time.time()

# Training Loop
for e in range(EPOCHS):
    # Training
    model.train()
    trainRunningLoss = 0

    for i, (_, _, classVec, imgT) in enumerate(trainLoader):
        # print(f"shape imgT: {imgT.shape}")
        # print(f"epoch: {e}, batch: [{i}/{len(trainLoader)}]")
        imgT = imgT.to(device)
        classVec = classVec.to(device)
        preds = model(imgT)

        loss = lossFn(preds, classVec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainRunningLoss += loss.item()

    # Eval
    model.eval()

    avrgTrainLoss = trainRunningLoss / len(trainLoader)

    perEpochOutputDir = os.path.join(
        OUTPUT_DIR, f"epoch{e:02d}"
    )
    os.makedirs(perEpochOutputDir, exist_ok=False)

    # create subfolders for predicated to groudn truth for eval
    for predC in list(Classes):
        for trueC in list(Classes):
            os.makedirs(os.path.join(
                perEpochOutputDir, f"P_{predC}_T_{trueC}"), exist_ok=False)

    valLoss = 0

    torch.save(model.state_dict(), os.path.join(
        MODELS_OUTPUT_DIR, f"cnn_e{e:02d}.pth"))

    with torch.no_grad():
        for filePath, classNumber, classVec, imgT in validateLoader:
            imgT = imgT.to(device)
            classVec = classVec.to(device)

            preds = model(imgT)

            loss = lossFn(preds, classVec)
            valLoss += loss.item()

            pred_classes = torch.argmax(preds, dim=1)

            for i, pred_class in enumerate(pred_classes):
                trueC = Classes(classNumber[i].item())
                predC = Classes(pred_class.item() + 1)

                filename = os.path.basename(filePath[i])

                dirToSave = os.path.join(
                    perEpochOutputDir, f"P_{predC}_T_{trueC}")

                fileToSave = os.path.join(dirToSave, filename)

                shutil.copyfile(filePath[i], fileToSave)

    avrgValLoss = valLoss / len(validateLoader)

    runtime = time.time() - startTime

    with open(STATS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([e, avrgTrainLoss, avrgValLoss, runtime])

    print(f"Epoch [{e} / {EPOCHS - 1}] "
          f"Train loss {avrgTrainLoss} "
          f"Val loss {avrgValLoss} "
          f"runtime {runtime}")
