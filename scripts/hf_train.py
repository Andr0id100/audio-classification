from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)
from torchaudio.transforms import Resample
import torch
from utils import get_dataset, compute_metrics

dataset = get_dataset()

classes = sorted(list(set(dataset["train"]["class"])))
id2label = {i: classes[i] for i in range(len(classes))}
label2id = {classes[i]: i for i in range(len(classes))}

num_labels = len(classes)


# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# model = AutoModelForAudioClassification.from_pretrained(
#     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
# )
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-16-16-0.442"
)
model = AutoModelForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-16-16-0.442",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# for param in model.base_model.parameters():
#     param.requires_grad = False

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


training_args = TrainingArguments(
    output_dir="mit-30",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    logging_steps=10,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
