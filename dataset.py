import json
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class PDMLiteDataset(Dataset):
    def __init__(self, path: str):
        self.path: str = path

        self.paths: list[str] = self.__load_paths()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __load_paths(self) -> list[str]:
        base_paths: list[str] = list(sorted(os.listdir(self.path)))

        paths: list[str] = list()

        for base_path in base_paths:
            route_town_path: list[str] = list(sorted(os.listdir(os.path.join(self.path, base_path))))
            for path in route_town_path:
                if not os.path.exists(os.path.join(self.path, base_path, path, "rgb")):
                    continue

                steps: list[str] = list(sorted(os.listdir(os.path.join(self.path, base_path, path, "rgb"))))
                for step in steps:
                    full_path = os.path.join(self.path, base_path, path, "rgb", step)
                    paths.append(full_path)
        
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = Image.open(path)
        image = np.array(image)
        image = self.transform(image)
        image = image.unsqueeze(0)

        measurements_path = path.replace("rgb", "measurements").replace("jpg", "json")
        
        with open(measurements_path) as f:
            measurements = json.load(f)
        
        route = measurements["route"]

        for i in range(20):
            if i % 2 == 0:
                route.append(route[i])

        route = torch.tensor(route, dtype=torch.float32)

        return image, route
