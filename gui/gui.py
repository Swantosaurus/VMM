import gradio as gr
from PIL import Image
import torch
import cv2 as cv
import numpy as np
import time
import os
import random
from captum.attr import Saliency, IntegratedGradients, DeepLift, GuidedGradCam
from LeafDataset import Classes, ClassMapper
from models.ownCNN.ConvolutionClassifierModel import ConvolutionClassifierModel128Drop, ConvolutionClassifierModel128NoDrop, ConvolutionClassifierModel256Drop, ConvolutionClassifierModel256NoDrop
from models.ownVisualTransformer.VisualTransformerModel import VisualTransformer
from vit_grad_rollout import VITAttentionGradRollout
from transformers import AutoImageProcessor, ResNetConfig, ResNetForImageClassification


TOP_K = 3

print("Loading models to memory")

resNetProcessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
resNetConfig = ResNetConfig.from_pretrained("microsoft/resnet-50", num_labels=len(Classes))

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.mps.is_available else device

print(f"device used: {device}")

classes = len(Classes)

modelCNN128Lr3NoDrop = ConvolutionClassifierModel128NoDrop(
    classes).to(device).eval()
modelCNN128Lr4NoDrop = ConvolutionClassifierModel128NoDrop(
    classes).to(device).eval()
modelCNN128Lr5NoDrop = ConvolutionClassifierModel128NoDrop(
    classes).to(device).eval()

modelCNN128Lr3Drop = ConvolutionClassifierModel128Drop(
    classes).to(device).eval()
modelCNN128Lr4Drop = ConvolutionClassifierModel128Drop(
    classes).to(device).eval()
modelCNN128Lr5Drop = ConvolutionClassifierModel128Drop(
    classes).to(device).eval()

modelCNN256Lr3NoDrop = ConvolutionClassifierModel256NoDrop(
    classes).to(device).eval()
modelCNN256Lr3Drop = ConvolutionClassifierModel256Drop(
    classes).to(device).eval()

modelViT = VisualTransformer(classes).to(device).eval()


class HFResNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        return out.logits

modelResNet128 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", config=resNetConfig, ignore_mismatched_sizes=True).to(device).eval()
modelResNet256 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", config=resNetConfig, ignore_mismatched_sizes=True).to(device).eval()


print("loading states dict to memory")
stateDic128Lr3NoDrop = torch.load(
    "./models/ownCNN/128NoDropoutLr3.pth", map_location=device)
stateDic128Lr4NoDrop = torch.load(
    "./models/ownCNN/128NoDropoutLr4.pth", map_location=device)
stateDic128Lr5NoDrop = torch.load(
    "./models/ownCNN/128NoDropoutLr5.pth", map_location=device)

stateDic128Lr3Drop = torch.load(
    "./models/ownCNN/128DropoutLr3.pth", map_location=device)
stateDic128Lr4Drop = torch.load(
    "./models/ownCNN/128DropoutLr4.pth", map_location=device)
stateDic128Lr5Drop = torch.load(
    "./models/ownCNN/128DropoutLr5.pth", map_location=device)

stateDic256Lr3NoDrop = torch.load(
    "./models/ownCNN/256NoDropoutLr3.pth", map_location=device)
stateDic256Lr3Drop = torch.load(
    "./models/ownCNN/256DropoutLr3.pth", map_location=device)

stateDicVit128Lr3 = torch.load(
    "./models/ownVisualTransformer/128Lr3.pth", map_location=device)

stateDic128ResNet = torch.load(
    "./models/ResNet50tl/128Lr3.pth"
)

stateDick256ResNet = torch.load(
    "./models/ResNet50tl/256Lr3.pth"
)

print("attachign states to models")
modelCNN128Lr3NoDrop.load_state_dict(stateDic128Lr3NoDrop)
modelCNN128Lr4NoDrop.load_state_dict(stateDic128Lr4NoDrop)
modelCNN128Lr5NoDrop.load_state_dict(stateDic128Lr5NoDrop)

modelCNN128Lr3Drop.load_state_dict(stateDic128Lr3Drop)
modelCNN128Lr4Drop.load_state_dict(stateDic128Lr4Drop)
modelCNN128Lr5Drop.load_state_dict(stateDic128Lr5Drop)

modelCNN256Lr3NoDrop.load_state_dict(stateDic256Lr3NoDrop)
modelCNN256Lr3Drop.load_state_dict(stateDic256Lr3Drop)

modelViT.load_state_dict(stateDicVit128Lr3)


modelResNet128.load_state_dict(stateDic128ResNet)
modelResNet256.load_state_dict(stateDick256ResNet)

modelResNet128 = HFResNetWrapper(modelResNet128)
modelResNet256 = HFResNetWrapper(modelResNet256)

models = {
    "CNN128": {
        "CNN128Lr3NoDrop": modelCNN128Lr3NoDrop,
        "CNN128Lr4NoDrop": modelCNN128Lr4NoDrop,
        "CNN128Lr5NoDrop": modelCNN128Lr5NoDrop,

        "CNN128Lr3Drop": modelCNN128Lr3Drop,
        "CNN128Lr4Drop": modelCNN128Lr4Drop,
        "CNN128Lr5Drop": modelCNN128Lr5Drop
    },
    "CNN256": {
        "CNN256Lr3NoDrop": modelCNN256Lr3NoDrop,
        "CNN256Lr3Drop": modelCNN256Lr3Drop
    },
    "ResNet128": {
        "ResNet128TlClas": modelResNet128
    },
    "ResNet256": {
        "ResNet256TlClas": modelResNet256
    },
    "ViT128": {
        "ViT128l5": modelViT
    }
}

modelNames = [key for group in models.values() for key in group.keys()]


def imgToTensor(img, imgSize):
    if img is None:
        raise RuntimeError("img is missing")
    if imgSize is None:
        raise RuntimeError("size is missing")

    img = np.array(img)
    img = cv.resize(img, (imgSize, imgSize), interpolation=cv.INTER_AREA)
    img = img / 255.0

    imgT = torch.from_numpy(img).permute(2, 0, 1).float()
    return imgT.to(device).unsqueeze(0)


def indentLeft(str, n):
    out = ""
    for _ in range(n):
        out += "    "
    return out + str


def classify_image(image: Image.Image):
    """Accepts PIL image, runs through your model, returns label."""

    img128t = imgToTensor(image, 128)
    img256t = imgToTensor(image, 256)

    output = ""
    for groupName, groups in models.items():
        output += indentLeft(f"## {groupName} MODELS:\n", 0)

        imgT = img128t if "128" in groupName else img256t

        for modelName, model in groups.items():
            startTime = time.time()
            preds = model(imgT)
            evalTime = time.time() - startTime

            probs = torch.softmax(preds, dim=1)
            top3_conf, top3_idx = torch.topk(probs, k=TOP_K, dim=1)
            top3_conf = top3_conf.squeeze(0).tolist()
            top3_idx = top3_idx.squeeze(0).tolist()

            output += indentLeft(f"#### {modelName} in {evalTime:.4f}\n", 0)
            for i in range(TOP_K):
                output += indentLeft(f"{i + 1}. {Classes(top3_idx[i]).name} with {
                                     top3_conf[i]:.4f}\n", 2)

    return output


cm = ClassMapper("../dataset/")

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def loadLocalImageClass(label):
    if label is None:
        label = Classes(0).name

    c = Classes[label]
    path = cm.classToPath(c)

    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    random.shuffle(files)

    files = files[:20]

    return [Image.open(img) for img in files]


def findModel(modelNameToFind):
    for modelMappings in models.values():
        for modelName, model in modelMappings.items():
            if modelName == modelNameToFind:
                return model


def loadImagesForClasses(classesIdx):
    classes = [Classes(i) for i in classesIdx]

    dirs = [cm.classToPath(c) for c in classes]

    images = []

    for dir in dirs:
        files = [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        random.shuffle(files)
        files = files[:20]
        images.append([Image.open(img) for img in files])
    return images


def resize(img, size):
    if img is None:
        raise RuntimeError("no image")

    img = np.array(img)
    img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)

    return img


def calculateInterpretability(img, imgT, model, imgSize, targetClass):
    imgToInpaint = resize(img, imgSize)
    imgToInpaint = cv.cvtColor(imgToInpaint, cv.COLOR_RGB2GRAY)
    imgToInpaint = imgToInpaint / 255.0 / 2
    imgToInpaint = np.stack([imgToInpaint]*3, axis=-1)

    imgSal = imgToInpaint.copy()
    imgIg = imgToInpaint.copy()

    saliancy = Saliency(model)
    ig = IntegratedGradients(model)

    saliancyAttr = saliancy.attribute(imgT, target=targetClass.value)
    igAttr = ig.attribute(imgT, target=targetClass.value)

    saliancyAttr = saliancyAttr.detach().cpu().numpy()
    igAttr = igAttr.detach().cpu().numpy()

    saliancyAttr = saliancyAttr.squeeze(0)
    igAttr = igAttr.squeeze(0)

    if saliancyAttr.ndim == 3:
        saliancyAttr = saliancyAttr.mean(axis=0)
    if igAttr.ndim == 3:
        igAttr = igAttr.mean(axis=0)

    saliancyAttr = np.abs(saliancyAttr)
    igAttr = np.abs(igAttr)

    saliancyAttr = saliancyAttr / saliancyAttr.max()
    igAttr = igAttr / igAttr.max()

    imgSal[..., 2] = np.clip(imgToInpaint[..., 2] + saliancyAttr * 1, 0, 1)
    imgIg[..., 2] = np.clip(imgToInpaint[..., 2] + igAttr * 1, 0, 1)

    imgSal = Image.fromarray((imgSal * 255).astype(np.uint8))
    imgIg = Image.fromarray((imgIg * 255).astype(np.uint8))

    imgSal = resize(imgSal, 256)
    imgIg = resize(imgIg, 256)

    return [imgSal, imgIg]


def calculateInterpretabilityViT(img, imgT, model, imgSize, targetClass):
    imgToInpaint = resize(img, imgSize)
    imgToInpaint = cv.cvtColor(imgToInpaint, cv.COLOR_RGB2GRAY)
    imgToInpaint = imgToInpaint / 255.0 / 2
    imgToInpaint = np.stack([imgToInpaint]*3, axis=-1)

    print(f"targetClass {targetClass}")

    grad_rollout = VITAttentionGradRollout(
        model, device, discard_ratio=0.9, attention_layer_name="att")
    mask = grad_rollout(imgT, category_index=targetClass.value)

    # mask is size of patches not img
    mask = cv.resize(mask, (imgSize, imgSize), interpolation=cv.INTER_CUBIC)
    mask = mask / mask.max()

    masked = imgToInpaint.copy()
    masked[..., 2] = np.clip(imgToInpaint[..., 2] + mask, 0, 1)

    masked = Image.fromarray((masked * 255).astype(np.uint8))
    masked = resize(masked, 256)
    return masked


def classifyImageWithModel(img, model):
    modelToUse = findModel(model)
    size = 128 if "128" in model else 256

    output = ""

    imgT = imgToTensor(img, size)

    startTime = time.time()
    preds = modelToUse(imgT)
    evalTime = time.time() - startTime

    probs = torch.softmax(preds, dim=1)
    top3_conf, top3_idx = torch.topk(probs, k=TOP_K, dim=1)
    top3_conf = top3_conf.squeeze(0).tolist()
    top3_idx = top3_idx.squeeze(0).tolist()

    output += indentLeft(f"#### {model} in {evalTime:.4f}\n", 0)
    for i in range(TOP_K):
        output += indentLeft(f" {i + 1}. {Classes(top3_idx[i]).name} with {
            top3_conf[i]:.4f}\n", 1)

    imagesTop1, imagesTop2, imagesTop3 = loadImagesForClasses(top3_idx)

    if "ViT" in model:
        saliancy1 = calculateInterpretabilityViT(
            img, imgT, modelToUse, size, Classes(top3_idx[0]))
        integrated_gradients1 = saliancy1

        saliancy2 = calculateInterpretabilityViT(
            img, imgT, modelToUse, size, Classes(top3_idx[1]))
        integrated_gradients2 = saliancy2

        saliancy3 = calculateInterpretabilityViT(
            img, imgT, modelToUse, size, Classes(top3_idx[2]))
        integrated_gradients3 = saliancy3

    else:
        saliancy1, integrated_gradients1 = calculateInterpretability(
            img, imgT, modelToUse, size, Classes(top3_idx[0]))
        saliancy2, integrated_gradients2 = calculateInterpretability(
            img, imgT, modelToUse, size, Classes(top3_idx[1]))
        saliancy3, integrated_gradients3 = calculateInterpretability(
            img, imgT, modelToUse, size, Classes(top3_idx[2]))

    return [output, imagesTop1, imagesTop2, imagesTop3,
            saliancy1, integrated_gradients1,
            saliancy2, integrated_gradients2,
            saliancy3, integrated_gradients3
            ]

    # -----------------------------
    # Gradio UI
    # -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Image Classification")

    with gr.Tabs():
        with gr.Tab("AllModels"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Image")
                    classify_btn = gr.Button("Classify")

                output_label = gr.Markdown(label="Prediction")
                classify_btn.click(fn=classify_image, inputs=input_img,
                                   outputs=output_label)
                with gr.Column():
                    dd = gr.Dropdown(
                        choices=[c.name for c in list(Classes)],
                        multiselect=False,
                        label="Choise img class to display"
                    )
                    list_images = gr.Gallery()

                    dd.change(fn=loadLocalImageClass,
                              inputs=dd, outputs=list_images)

                    demo.load(loadLocalImageClass, None, list_images)
        with gr.Tab("Single model"):
            with gr.Row():

                with gr.Column():
                    input_img2 = gr.Image(type="pil", label="Upload Image")
                    modelDD = gr.Dropdown(
                        choices=modelNames,
                        multiselect=False,
                        label="Select Model to predict"
                    )

                    classify_btn2 = gr.Button("Classify")

                with gr.Column():

                    model_output2 = gr.Markdown()

                    with gr.Tabs():
                        with gr.Tab("Top 1"):
                            with gr.Column():
                                gr.Markdown("## Interpretability")
                                with gr.Row():
                                    saliancy1 = gr.Image(
                                        type="pil", label="Saliancy")
                                    integrated_gradients1 = gr.Image(
                                        type="pil", label="Integrated Gradients")
                            gr.Markdown("## Images From same class")
                            model_output_galery1 = gr.Gallery()
                        with gr.Tab("Top 2"):
                            with gr.Column():
                                gr.Markdown("## Interpretability")
                                with gr.Row():
                                    saliancy2 = gr.Image(
                                        type="pil", label="Saliancy")
                                    integrated_gradients2 = gr.Image(
                                        type="pil", label="Integrated Gradients")
                            gr.Markdown("## Images From same class")
                            model_output_galery2 = gr.Gallery()
                        with gr.Tab("Top 3"):
                            with gr.Column():
                                gr.Markdown("## Interpretability")
                                with gr.Row():
                                    saliancy3 = gr.Image(
                                        type="pil", label="Saliancy")
                                    integrated_gradients3 = gr.Image(
                                        type="pil", label="Integrated Gradients")
                            gr.Markdown("## Images From same class")
                            model_output_galery3 = gr.Gallery()

            classify_btn2.click(
                fn=classifyImageWithModel,
                inputs=[input_img2, modelDD],
                outputs=[model_output2, model_output_galery1,
                         model_output_galery2, model_output_galery3,
                         saliancy1, integrated_gradients1,
                         saliancy2, integrated_gradients2,
                         saliancy3, integrated_gradients3
                         ]
            )

# Launch app
if __name__ == "__main__":
    demo.launch()
