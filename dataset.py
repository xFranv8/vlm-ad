
import os

from utils import load_llama_tokenizer

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class LingoQA(Dataset):
    def __init__(self, dataframe: pd.DataFrame, segments_path: str):
        self.path: str = segments_path

        self.dataframe: pd.DataFrame = dataframe

        self.tokenizer = load_llama_tokenizer()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        segment: str = self.dataframe["segment_id"].iloc[idx]
        questions: list = self.dataframe[self.dataframe["segment_id"] == segment]["question"].tolist()
        questions_ids: list = self.dataframe[self.dataframe["segment_id"] == segment]["question_id"].tolist()
        answers: list = list()

        for id in questions_ids:
            answers.append(self.dataframe[self.dataframe["question_id"] == id]["answer"].iloc[0])
        
        images: list[np.ndarray] = list()
        for i in range(5):
            image: np.array = np.array(Image.open(os.path.join(self.path, f"{segment}/{i}.jpg")))
            image = torch.from_numpy(image)
            images.append(image)
        
        images = torch.stack(images)
        
        rand_idx = np.random.randint(0, len(questions))

        question = questions[rand_idx]
        answer = answers[rand_idx]

        question = self.tokenizer(question, return_tensors="pt")
        question_ids = question.input_ids

        answer = self.tokenizer(answer, return_tensors="pt")
        answer_ids = answer.input_ids

        input_ids = torch.cat((question_ids, answer_ids), dim=1)
        label_ids = input_ids.clone()

        return images, input_ids, label_ids
    