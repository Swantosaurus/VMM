import cv2 as cv
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from enum import Enum


class Classes(Enum):
    # Apple
    APPLE_HEALTHY = 0
    APPLE_CEDAR_APPLE_RUST = 1
    APPLE_BLACK_ROT = 2
    APPLE_SCAB = 3

    # Cherry
    CHERRY_HEALTHY = 4
    CHERRY_POWDERY_MILDEW = 5

    # Grape
    GRAPE_HEALTHY = 6
    GRAPE_BLACK_ROT = 7
    GRAPE_ESCA = 8
    GRAPE_LEAF_BLIGHT = 9

    # Peach
    PEACH_HEALTHY = 10
    PEACH_BACTERIAL_SPOT = 11

    # Pepper
    PEPPER_HEALTHY = 12
    PEPPER_BACTERIAL_SPOT = 13

    # Potato
    POTATO_HEALTHY = 14
    POTATO_EARLY_BLIGHT = 15
    POTATO_LATE_BLIGHT = 16

    # Strawberry
    STRAWBERRY_HEALTHY = 17
    STRAWBERRY_LEAF_SCORCH = 18

    # Tomato
    TOMATO_HEALTHY = 19
    TOMATO_EARLY_BLIGHT = 20
    TOMATO_LATE_BLIGHT = 21
    TOMATO_LEAF_MOLD = 22
    TOMATO_BACTERIAL_SPOT = 23
    TOMATO_SEPTORIA_LEAF_SPOT = 24
    TOMATO_SPIDER_MITE = 25
    TOMATO_TARGET_SPOT = 26
    TOMATO_MOSAIC_VIRUS = 27
    TOMATO_YELLOW_LEAF_CURL_VIRUS = 28


class ClassMapper():
    def __init__(self, relativePathToDataset):
        self.base = relativePathToDataset

        # Build maps
        self.class_to_path = {}
        self.path_to_class = {}

        # Apple
        self.class_to_path[Classes.APPLE_HEALTHY] = os.path.join(
            self.base, "Apple___healthy")
        self.class_to_path[Classes.APPLE_CEDAR_APPLE_RUST] = os.path.join(
            self.base, "Apple___Cedar_apple_rust")
        self.class_to_path[Classes.APPLE_BLACK_ROT] = os.path.join(
            self.base, "Apple___Black_rot")
        self.class_to_path[Classes.APPLE_SCAB] = os.path.join(
            self.base, "Apple___Apple_scab")

        # Cherry
        self.class_to_path[Classes.CHERRY_HEALTHY] = os.path.join(
            self.base, "Cherry___healthy")
        self.class_to_path[Classes.CHERRY_POWDERY_MILDEW] = os.path.join(
            self.base, "Cherry___Powdery_mildew")

        # Grape
        self.class_to_path[Classes.GRAPE_HEALTHY] = os.path.join(
            self.base, "Grape___healthy")
        self.class_to_path[Classes.GRAPE_BLACK_ROT] = os.path.join(
            self.base, "Grape___Black_rot")
        self.class_to_path[Classes.GRAPE_ESCA] = os.path.join(
            self.base, "Grape___Esca_(Black_Measles)")
        self.class_to_path[Classes.GRAPE_LEAF_BLIGHT] = os.path.join(
            self.base, "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)")

        # Peach
        self.class_to_path[Classes.PEACH_HEALTHY] = os.path.join(
            self.base, "Peach___healthy")
        self.class_to_path[Classes.PEACH_BACTERIAL_SPOT] = os.path.join(
            self.base, "Peach___Bacterial_spot")

        # Pepper
        self.class_to_path[Classes.PEPPER_HEALTHY] = os.path.join(
            self.base, "Pepper,_bell___healthy")
        self.class_to_path[Classes.PEPPER_BACTERIAL_SPOT] = os.path.join(
            self.base, "Pepper,_bell___Bacterial_spot")

        # Potato
        self.class_to_path[Classes.POTATO_HEALTHY] = os.path.join(
            self.base, "Potato___healthy")
        self.class_to_path[Classes.POTATO_EARLY_BLIGHT] = os.path.join(
            self.base, "Potato___Early_blight")
        self.class_to_path[Classes.POTATO_LATE_BLIGHT] = os.path.join(
            self.base, "Potato___Late_blight")

        # Strawberry
        self.class_to_path[Classes.STRAWBERRY_HEALTHY] = os.path.join(
            self.base, "Strawberry___healthy")
        self.class_to_path[Classes.STRAWBERRY_LEAF_SCORCH] = os.path.join(
            self.base, "Strawberry___Leaf_scorch")

        # Tomato
        self.class_to_path[Classes.TOMATO_HEALTHY] = os.path.join(
            self.base, "Tomato___healthy")
        self.class_to_path[Classes.TOMATO_EARLY_BLIGHT] = os.path.join(
            self.base, "Tomato___Early_blight")
        self.class_to_path[Classes.TOMATO_LATE_BLIGHT] = os.path.join(
            self.base, "Tomato___Late_blight")
        self.class_to_path[Classes.TOMATO_BACTERIAL_SPOT] = os.path.join(
            self.base, "Tomato___Bacterial_spot")
        self.class_to_path[Classes.TOMATO_LEAF_MOLD] = os.path.join(
            self.base, "Tomato___Leaf_Mold")
        self.class_to_path[Classes.TOMATO_SEPTORIA_LEAF_SPOT] = os.path.join(
            self.base, "Tomato___Septoria_leaf_spot")
        self.class_to_path[Classes.TOMATO_SPIDER_MITE] = os.path.join(
            self.base, "Tomato___Spider_mites Two-spotted_spider_mite")
        self.class_to_path[Classes.TOMATO_TARGET_SPOT] = os.path.join(
            self.base, "Tomato___Target_Spot")
        self.class_to_path[Classes.TOMATO_MOSAIC_VIRUS] = os.path.join(
            self.base, "Tomato___Tomato_mosaic_virus")
        self.class_to_path[Classes.TOMATO_YELLOW_LEAF_CURL_VIRUS] = os.path.join(
            self.base, "Tomato___Tomato_Yellow_Leaf_Curl_Virus")

        for clas, path in self.class_to_path.items():
            self.path_to_class[path] = clas

    def allClasses(self):
        return list(Classes)

    def allPaths(self):
        return self.class_to_path.values()

    def allClassToPathsMappings(self):
        return self.class_to_path

    def allPathToClassMappings(self):
        return self.path_to_class

    def pathToClass(self, path):
        return self.path_to_class[path]

    def classToPath(self, clas):
        return self.class_to_path[clas]


class LeafDataset(Dataset):
    def __init__(self, datasetDir, nFromFile, nToFile):
        self.mapper = ClassMapper(datasetDir)

        if nFromFile > nToFile:
            raise AttributeError(f"{nFromFile} nFromFile > nToFile {nToFile}")

        self.nFromFile = nFromFile
        self.nToFile = nToFile
        self.fileCountPerClass = nToFile - nFromFile

        self.numClasses = len(Classes)

        self.cacheListDirs = {}
        for c in list(Classes):
            dir_path = self.mapper.classToPath(c)
            files = sorted(os.listdir(dir_path))
            self.cacheListDirs[c] = files

    def __len__(self):
        le = len(self.mapper.allClasses()) * self.fileCountPerClass
        return le

    def getClassFromIndex(self, idx):
        classId = (idx // self.fileCountPerClass)
        return Classes(classId)

    def getImagePositionFromIndex(self, idx):
        return idx % self.fileCountPerClass + self.nFromFile

    def __getitem__(self, idx):
        clas = self.getClassFromIndex(idx)
        imagePos = self.getImagePositionFromIndex(idx)
        dir = self.mapper.classToPath(clas)
        listDir = self.cacheListDirs[clas]

        path = os.path.join(dir, listDir[imagePos])

        img = cv.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        peak = np.zeros(self.numClasses, dtype=np.float32)
        peak[clas.value] = 1.0

        return path, clas.value, peak, img_tensor
