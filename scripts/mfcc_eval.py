from utils import get_dataset
from torchaudio.transforms import MFCC
import torch
import pandas as pd
from tqdm import tqdm

# Hyperparameters
NUM_EPOCHS = 20
N_MFCC = 40

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


model = torch.load("../models/mfcc_model.pt")

model.eval()

data = []
for i in tqdm(range(len(encoded_dataset["test"]))):
    inputs = torch.tensor(encoded_dataset["test"][i]["mfcc"])
    preds = model(inputs).softmax(dim=0)

    data.append(
        (
            encoded_dataset["test"][i]["slice_file_name"],
            encoded_dataset["test"][i]["class"],
            *preds.detach().numpy(),
        )
    )
# import pdb; pdb.set_trace()

df = pd.DataFrame(data, columns=["filename", "label"] + classes)
df.to_csv("mfcc-test.csv", index=False)
