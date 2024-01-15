from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate

def get_dataset():
    dataset = load_dataset("danavery/urbansound8K")["train"]

    # Used seed parameter to ensure consistency in results
    train_test_valid = dataset.train_test_split(test_size=0.3, seed=0)
    test_valid = train_test_valid['test'].train_test_split(test_size=0.5, seed=0)

    return DatasetDict({
        'train': train_test_valid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
    })

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)