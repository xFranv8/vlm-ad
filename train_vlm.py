from dataset import PDMLiteDataset
from vlm import load_model, BugaVLM

import os

from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.utils.data import DataLoader
import torch
import pandas as pd
import wandb


def train(data_loader: DataLoader, lr: float, num_epochs: int) -> BugaVLM:
    wandb.init(project="vlms",
               config={"lr": lr, "num_epochs": 1}
               )

    model = load_model()
    model = model.to("cuda:0")

    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(data_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    step: int = 0

    model.train()
    for epoch in range(num_epochs):
        for images, route in data_loader:
            images = images.to("cuda:0")
            route = route.to("cuda:0")

            optimizer.zero_grad()
            
            predicted_route = model(images=images)
            loss = loss_fn(predicted_route, route)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.update(1)

            wandb.log({"loss": loss.item(), "step": step})
            
            step += 1
        
        wandb.log({"epoch": epoch})
    
    return model


def main() -> None:
    dataset: PDMLiteDataset = PDMLiteDataset("data")
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model: BugaVLM = train(data_loader, lr=5e-6, num_epochs=10)

    clip_model = model.clip_model
    llama_model = model.llama_model
    project_model = model.llama_proj

    clip_model.push_to_hub("clip-vit-base-patch32")
    llama_model.push_to_hub("llama-1.1B-Chat-v1.0")
    torch.save(project_model.state_dict(), "project_model.pt")

if __name__ == "__main__":
    main()
