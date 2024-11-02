
import string
import argparse
import pathlib
import pickle
import random
import accelerate
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='facebook/opt-13b')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
seed_value = 10

if 'opt' in args.model:
    model_name = 'opt'
elif 'llama' in args.model:
    model_name = 'llama'
elif 'gpt-neo' in args.model:
    model_name = 'gpt-neo'
elif 'phi' in args.model:
    model_name = 'phi'
elif 'gemma' in args.model:
    model_name = 'gemma'
else:
    raise NotImplementedError

# Use pqa-l for labeled data
if not pathlib.Path(f'{config.data_dir}/pub_med_qa_{model_name}').exists():
    print('Preprocessing dataset')
    dataset = datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    
    # No val split, so split the train portion into train and val
    train_val_split = dataset['train'].train_test_split(test_size=0.2)
    train_data, val_data = train_val_split['train'], train_val_split['test']
    data_for_few_shot_prompt = train_data.select(range(0, 5))

    few_shot_prompt = 'This is a bot that correctly answers questions. \n'
    for sample in data_for_few_shot_prompt:
        few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['final_decision'] + ' '

    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        answers = [answer for answer in batch["final_decision"]]

        batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)
        batch['question_id'] = [''.join(random.choices(string.ascii_uppercase + string.digits, k=32)) for _ in batch['question']]
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch['answer'] = answers
        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]
        return batch
    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["context", "long_answer"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "decoder_input_ids", "attention_mask", "decoder_attention_mask"],
        output_all_columns=True)

    val_data.save_to_disk(f'{config.data_dir}/pub_med_qa_{model_name}')
else:
    val_data = datasets.load_from_disk(f'{config.data_dir}/pub_med_qa_{model_name}')
