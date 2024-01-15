from utils import get_dataset
from torchaudio.transforms import MFCC
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
from model import MFCCModel


# Hyperparameters
NUM_EPOCHS = 20
N_MFCC = 40

run = wandb.init(
    project="augnito-mfcc",
)

dataset = get_dataset()
classes = sorted(list(set(dataset["train"]["class"])))
SAMPLE_RATE = dataset["train"][0]["audio"]["sampling_rate"]

mfcc_transform = MFCC(n_mfcc=N_MFCC, sample_rate=SAMPLE_RATE)


def preprocess_function(datapoint):
    audio = torch.tensor(datapoint["audio"]["array"], dtype=torch.float)
    result = {
        "mfcc": mfcc_transform(audio).mean(-1),
    }
    return result


encoded_dataset = dataset.map(preprocess_function, remove_columns="audio")


model = MFCCModel(N_MFCC, len(classes))
optimizer = torch.optim.Adam(model.parameters())

accumulated_loss = 0
for epoch in tqdm(range(NUM_EPOCHS)):
    model.train()
    for i, x in enumerate(encoded_dataset["train"]):
        inputs, labels = torch.tensor(x["mfcc"]), torch.tensor(x["classID"])
        optimizer.zero_grad()
        preds = model(inputs)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.item()
        if i % 100 == 99:  # Log every 100 steps
            wandb.log({"train/loss": accumulated_loss / 100})
            accumulated_loss = 0

    model.eval()
    final_preds = []
    for x in encoded_dataset["valid"]:
        inputs = torch.tensor(x["mfcc"])
        preds = model(inputs)
        final_preds.append(preds.argmax().item())

    final_preds = np.array(final_preds)
    wandb.log(
        {
            "eval/accuracy": (final_preds == encoded_dataset["valid"]["classID"]).sum()
            / len(encoded_dataset["valid"])
        }
    )

torch.save(model, "mfcc_model.pt")