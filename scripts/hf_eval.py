from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from utils import get_dataset
from torchaudio.transforms import Resample
import torch
from tqdm import tqdm
import pandas as pd

dataset = get_dataset()


classes = sorted(list(set(dataset["train"]["class"])))
id2label = {i: classes[i] for i in range(len(classes))}
label2id = {classes[i]: i for i in range(len(classes))}

num_labels = len(classes)

# MODEL_PATH = "models/wav2vec2-finetuned"
MODEL_PATH = "models/mit-finetuned"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)


resampler = Resample(
    dataset["train"][0]["audio"]["sampling_rate"], feature_extractor.sampling_rate
)

def preprocess_function(examples):
    audio_arrays = [
        resampler(torch.tensor(x["array"], dtype=torch.float32)).numpy()
        for x in examples["audio"]
    ]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=32000,
        padding="longest",
        truncation=True,
    )
    return inputs


encoded_dataset = dataset.map(preprocess_function, remove_columns="audio", batched=True)
encoded_dataset = encoded_dataset.rename_column("classID", "label")

data = []
for i in tqdm(range(len(encoded_dataset["test"]))):
    input_values = torch.tensor(encoded_dataset["test"][i]["input_values"]).unsqueeze(0)
    preds = model(input_values).logits[0]
    preds = torch.softmax(preds, 0)

    data.append(
        (
            encoded_dataset["test"][i]["slice_file_name"],
            encoded_dataset["test"][i]["class"],
            *preds.detach().numpy(),
        )
    )

df = pd.DataFrame(
    data, columns=["filename", "label"] + list(model.config.label2id.keys())
)
df.to_csv("mit-test.csv", index=False)
