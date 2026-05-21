import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from model import ESPCN
from dataset import get_dataloader


class Trainer:

    def __init__(self, train_loader):
        self.device = Config.DEVICE
        self.epochs = Config.EPOCHS
        self.train_loader = train_loader

        self.model = ESPCN(upscale_factor=Config.UPSCALE_FACTOR).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        print(
            f"[INFO] Trainer initialized. Device: {self.device}, Epochs: {self.epochs}"
        )

    def train(self):
        print("\n--- STARTING TRAINING ---")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.model.train()

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"
            )

            for lr_imgs, hr_imgs in progress_bar:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)

                outputs = self.model(lr_imgs)
                loss = self.criterion(outputs, hr_imgs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"})

            avg_loss = epoch_loss / len(self.train_loader)
            print(
                f"Epoch [{epoch+1}/{self.epochs}] completed. Average Loss: {avg_loss:.6f}"
            )

        print("--- TRAINING COMPLETED ---")
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), Config.MODEL_PATH)
        print(f"[SUCCESS] Model saved successfully to {Config.MODEL_PATH}")
