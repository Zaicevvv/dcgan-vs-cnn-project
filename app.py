import streamlit as st
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import lightning.pytorch as pl
import torchmetrics
from torchvision import transforms, models
from tensorflow.keras.datasets import cifar10

(_, _), (x_test, _) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
class_labels = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

map_location = torch.device("cpu")

nz = 100
ngf = 64
num_channels = 3


class Gen_model(nn.Module):
    def __init__(self, ngpu):
        super(Gen_model, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


gen_model = torch.load("generator.pth", map_location)
gen_model.eval()


class LitModel(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log("train_acc", acc, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        self.log("test_acc", acc)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds


cnn_model = LitModel(num_classes=10)

checkpoint = torch.load("cnn_model.pth")
cnn_model.load_state_dict(checkpoint, strict=False)
cnn_model.eval()


def transform_image_for_cnn(image):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def transform_image_for_st(image):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image_resized = transform(image)
    image_np = image_resized.permute(0, 2, 3, 1).numpy()
    image_np = (image_np + 1) / 2
    image_np = 0.5 * image_np + 0.5
    return image_np


def get_random_cifar_image():
    idx = np.random.randint(0, len(x_test))
    image = transform_image_for_cnn(x_test[idx])
    return image, x_test[idx]


def generate_image():
    current_device = "cpu"
    fixed_noise = torch.randn(1, 100, 1, 1).to(current_device)

    with torch.no_grad():
        fake_image = gen_model(fixed_noise).detach().cpu()

    image_np = transform_image_for_st(fake_image)
    return fake_image, image_np


def show_cnn_prediction(image):
    with torch.no_grad():
        output = cnn_model(image)
        _, prediction = torch.max(output, 1)
        prediction = prediction.cpu().numpy()[0]

    predicted_label = class_labels[prediction]
    return predicted_label


st.header("Using a DCGAN Model to Combat Malicious CNN Agent as the Adversary")
st.title("CNN evaluation")

# Кнопки для дій
if st.button("Evaluate CNN with random CIFAR-10 image"):
    image, image_np = get_random_cifar_image()
    st.image(image_np, caption="Random CIFAR-10 image")
    prediction = show_cnn_prediction(image)
    st.write("CNN prediction: ", prediction)

st.title("DCGAN evaluation")

if st.button("Generate image with DCGAN and show CNN prediction"):
    generated_image, image_np = generate_image()
    st.image(image_np, caption="Generated image by DCGAN")
    prediction = show_cnn_prediction(generated_image)
    st.write("CNN prediction: ", prediction)
