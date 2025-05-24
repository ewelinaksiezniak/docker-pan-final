import argparse
import glob
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ===== Model =====
class ClassificationModel(nn.Module):
    def __init__(self, base_model_name="FacebookAI/xlm-roberta-base", hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, local_files_only = True)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    def load_weights(self, model_dir, device):
        safe_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.isfile(safe_path):
            state_dict = load_file(safe_path, device=device)
            print(f"✅ Wczytano wagi (safetensors): {safe_path}")
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location=device)
            print(f"✅ Wczytano wagi (.bin): {bin_path}")
        else:
            raise FileNotFoundError(f"❌ Brak wag w: {model_dir}")

        self.load_state_dict(state_dict, strict=False)

# ===== Dataset =====
class TextPairDataset(Dataset):
    def __init__(self, tokenizer, text1, text2, labels, max_length=512):
        self.tokenizer = tokenizer
        self.text1 = text1
        self.text2 = text2
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            str(self.text1[idx]),
            str(self.text2[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

def load_dataset(df, tokenizer, max_length=512):
    return TextPairDataset(
        tokenizer=tokenizer,
        text1=df["premise"].tolist(),
        text2=df["hypothesis"].tolist(),
        labels=df["labels"].tolist(),
        max_length=max_length
    )

def make_pairs(paragraphs):
    return [{"premise": paragraphs[i], "hypothesis": paragraphs[i+1], "labels": 0} for i in range(len(paragraphs) - 1)]

def load_paragraphs_from_file(path):
    with open(path, encoding="utf8") as f:
        return f.read().split('\n')

def predict_pairs(model, dataset, device):
    model.eval()
    preds = []
    dataloader = DataLoader(dataset, batch_size=16)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds

def main(args):

    model_dirs = {
        'easy':   "/root/.cache/huggingface/hub/models--Ewel--modeltrained_on_contrastive_encoder_10_epoch_quote_easy_freeze_0/snapshots/14bc749cb46c78ef096b46767cbc49ff0df5d03c",
        'medium': "/root/.cache/huggingface/hub/models--Ewel--model_trained_on_contrastive_encoder_10_epoch_question_medium_freeze_2/snapshots/82a87a9f1bbb17ee32833ae729502cd3c232e461",
        'hard':   "/root/.cache/huggingface/hub/models--Ewel--model_trained_on_contrastive_encoder_10_epoch_question_freeze_0/snapshots/07975659942d4a641854bb721123f06ca1ae27c2"
    }

    base_model_name = "FacebookAI/xlm-roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for difficulty in ['easy', 'medium', 'hard']:
        input_dir = os.path.join(args.input, difficulty)
        output_dir = os.path.join(args.output, difficulty)
        os.makedirs(output_dir, exist_ok=True)

        files_list = glob.glob(f'{input_dir}/**/*.txt', recursive=True)
        print(f'[{difficulty}] Znaleziono {len(files_list)} plików.')

        model = ClassificationModel(base_model_name=base_model_name)
        model.load_weights(model_dirs[difficulty], device=args.device)
        model.to(args.device)

        for file_path in tqdm(files_list):
            share_id = os.path.basename(file_path)[8:-4]
            paragraphs = load_paragraphs_from_file(file_path)
            pair_data = make_pairs(paragraphs)
            pair_df = pd.DataFrame(pair_data)
            dataset = load_dataset(pair_df, tokenizer)

            prediction = predict_pairs(model, dataset, args.device)
            result = {"changes": prediction}

            output_file = os.path.join(output_dir, f"solution-problem-{share_id}.json")
            with open(output_file, 'w', encoding='utf8') as fw:
                json.dump(result, fw, ensure_ascii=False, indent=2)

        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="../pan24_dataset/test", type=str)
    parser.add_argument("-o", "--output", default="../pan24_dataset/test", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("-e", "--enrich", action='store_true')
    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    print(f"use device {args.device}")

    main(args)
