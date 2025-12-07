from transformers import AutoImageProcessor, ResNetConfig, ResNetForImageClassification
from torch.utils.data import DataLoader
from LeafDataset import LeafDataset, Classes
import torch
import os
import shutil
import time
import csv


OUTPUT_DIR = "Resnt-01-256img"
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

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
config = ResNetConfig.from_pretrained("microsoft/resnet-50", num_labels=len(Classes))
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", config=config, ignore_mismatched_sizes=True).to(device)

# Freeze backbone
for param in model.base_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

startTime = time.time()

for e in range(EPOCHS):
    # Training
    model.train()
    trainingRunningLoss = 0

    for i, (_, classID, _, imgT) in enumerate(trainLoader):
        imgT = imgT.to(device)
        classID = classID
        classID = classID.to(device)

        preds = model(pixel_values=imgT, labels=classID)

        optimizer.zero_grad()
        loss = preds.loss
        loss.backward()
        optimizer.step()

        trainingRunningLoss += loss.item()

    avrgTrainLoss = trainingRunningLoss / len(trainLoader)

    # Eval
    model.eval()

    perEpochOutputDir = os.path.join(
        OUTPUT_DIR, f"epoch{e:02d}"
    )
    os.makedirs(perEpochOutputDir, exist_ok=False)

    # create subfolders for predicated togroudn truth for eval
    for predC in list(Classes):
        for trueC in list(Classes):
            os.makedirs(os.path.join(
                perEpochOutputDir, f"P_{predC}_T_{trueC}"), exist_ok=False)

    valLoss = 0

    torch.save(model.state_dict(), os.path.join(
        MODELS_OUTPUT_DIR, f"resnet50_e{e:02d}.pth"))

    with torch.no_grad():
        for filePath, classId, _, imgT in validateLoader:
            imgT = imgT.to(device)
            classId = classId
            classId = classId.to(device)

            preds = model(pixel_values=imgT, labels=classId)
            logits = preds.logits

            loss = preds.loss

            valLoss += loss.item()

            predClasses = torch.argmax(logits, dim=1)

            for i, predClass in enumerate(predClasses):
                predClassName = Classes(predClass.item() + 1)
                gtClass = Classes(classId[i].item() + 1)

                filename = os.path.basename(filePath[i])

                dirToSave = os.path.join(
                    perEpochOutputDir, f"P_{predClassName}_T_{gtClass}")

                fileToSave =os.path.join(dirToSave, filename)

                shutil.copyfile(filePath[i], fileToSave)

    avrValLoss = valLoss / len(validateLoader)

    runtime = time.time() - startTime

    with open(STATS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([e, avrgTrainLoss, avrValLoss, runtime])

    print(f"Epoch [{e} / {EPOCHS - 1}] "
          f"Train loss {avrgTrainLoss} "
          f"Val loss {avrValLoss} "
          f"runtime {runtime}")




