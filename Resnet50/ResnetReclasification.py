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


VALIDATE_FROM = 800
VALIDATE_TO = 1000

validateDataset = LeafDataset(INPUT_DATASET_DIR, VALIDATE_FROM, VALIDATE_TO)

validateLoader = DataLoader(validateDataset, batch_size=32)

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

model.eval()

startTime = time.time()

for e in range(EPOCHS):

    state_dict = torch.load(os.path.join(
        MODELS_OUTPUT_DIR, f"resnet50_e{e:02d}.pth"
    ), map_location=device)

    model.load_state_dict(state_dict)

    # evaluate only
    perEpochOutputDir = os.path.join(
        OUTPUT_DIR, f"epoch{e:02d}"
    )

    os.makedirs(perEpochOutputDir, exist_ok=False)

    # create subfolders for predicated togroudn truth for eval
    for predC in list(Classes):
        for trueC in list(Classes):
            os.makedirs(os.path.join(
                perEpochOutputDir, f"P_{predC}_T_{trueC}"), exist_ok=False)

    with torch.no_grad():
        for filePath, classId, _, imgT in validateLoader:
            imgT = imgT.to(device)
            classId = classId
            classId = classId.to(device)

            preds = model(pixel_values=imgT, labels=classId)
            logits = preds.logits

            predClasses = torch.argmax(logits, dim=1)

            for i, predClass in enumerate(predClasses):
                predClassName = Classes(predClass.item() + 1)
                gtClass = Classes(classId[i].item() + 1)

                filename = os.path.basename(filePath[i])

                dirToSave = os.path.join(
                    perEpochOutputDir, f"P_{predClassName}_T_{gtClass}")

                fileToSave =os.path.join(dirToSave, filename)

                shutil.copyfile(filePath[i], fileToSave)
    
    print(f"[{e}/{EPOCHS - 1}] time: {startTime - time.time()}")
    
    

    



