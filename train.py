from utils import load_model

from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.utils.data import DataLoader
from dataset import LingoQA
import torch
import pandas as pd


def collate_fn_pad(batch):
    input_ids = [input_ids for images, input_ids, label_ids in batch]
    label_ids = [label_ids for images, input_ids, label_ids in batch]

    max_len_input_ids = max([input_id.size(1) for input_id in input_ids])
    max_len_label_ids = max([label_id.size(1) for label_id in label_ids])

    max_len = max(max_len_input_ids, 384)

    padded_input_ids = [torch.nn.functional.pad(
        torch.tensor(ids), (0, max_len - ids.size(1)), value=0) for ids in input_ids]
    
    padded_label_ids = [torch.nn.functional.pad(
        torch.tensor(ids), (0, max_len - ids.size(1)), value=-100) for ids in label_ids]
    
    images = torch.stack([images for images, input_ids, label_ids in batch])
    
    tensor = torch.zeros((1, images.size(1)), dtype=torch.long)
    tensor *= -100
    padded_label_ids = [torch.cat((tensor, ids), dim=1) for ids in padded_label_ids]

    input_ids_batched = torch.stack(padded_input_ids)
    label_ids_batched = torch.stack(padded_label_ids)

    return images, input_ids_batched, label_ids_batched


df = pd.read_parquet("/workspace/Projects/vlms/LingoQA/train/train.parquet", engine='fastparquet')
df = df[:int(len(df) * 0.3)]
dataset = LingoQA(df, "/workspace/Projects/vlms/LingoQA/train/images/train")

data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_pad)

model = load_model()
model = model.to("cuda:1")

num_epochs = 20

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(data_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

step: int = 0

model.train()
for epoch in range(num_epochs):
    for images, input_ids, label_ids in data_loader:
        images = images.to("cuda:1")
        input_ids = input_ids.to("cuda:1")
        label_ids = label_ids.to("cuda:1")

        input_ids.long()
        label_ids.long()

        # concatenated_inputs = torch.Tensor().long().to("cuda:1")
        # for i in range(len(questions) - 1):
        #     tokenized_question = tokenizer(questions[i][0] + '\n', return_tensors="pt").to("cuda:1")
        #     tokenized_question_ids = tokenized_question.input_ids

        #     tokenized_answer = tokenizer(answers[i][0] + '\n\n', return_tensors="pt").to("cuda:1")
        #     tokenized_answer_ids = tokenized_answer.input_ids
        #     tokenized_answer_ids_with_filling = torch.cat((tensor, tokenized_answer.input_ids), dim=1)

        #     input_ids = torch.cat((tokenized_question_ids, tokenized_answer_ids), dim=1)
        #     concatenated_inputs = torch.cat((concatenated_inputs, input_ids), dim=1)

        # concatenated_labels = torch.cat((tensor, concatenated_inputs), dim=1)
        # concatenated_inputs.long()
        
        output = model.forward(images=images, input_ids=input_ids, labels=label_ids)
        loss = output["loss"]
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)

        if step % 100 == 0:
            print(f"Loss: {loss.item()}")
        
        step += 1

torch.save(model.state_dict(), "model.pt")
