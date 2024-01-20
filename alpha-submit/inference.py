import pandas as pd
from transformers import GPT2Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from datasets import Dataset

class CFG:
    tokenizer_name = '/tokenizer'
    model_good = '/model_good'
    model_brand = '/model_brand'
    device = 'cuda'
    batch_size = 32
    
def inference_good(test_ds, tokenizer):
    model = T5ForConditionalGeneration.from_pretrained(CFG.model_good, local_files_only=True)
    
    args = Seq2SeqTrainingArguments(
        'outputs/inference',
        per_device_train_batch_size = CFG.batch_size,
        per_device_eval_batch_size = CFG.batch_size,
        predict_with_generate = True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
    )
    
    test_preds = trainer.predict(test_ds)
    test_preds = tokenizer.batch_decode(test_preds.predictions, skip_special_tokens=True)
    test_pred_df = pd.DataFrame([[x.strip()] for x in test_preds], columns=['pred'])
    return test_pred_df['pred'].replace('нет', '')

def inference_brand(test_ds, tokenizer):
    model = T5ForConditionalGeneration.from_pretrained(CFG.model_brand, local_files_only=True)
    
    args = Seq2SeqTrainingArguments(
        'outputs/inference',
        per_device_train_batch_size = CFG.batch_size,
        per_device_eval_batch_size = CFG.batch_size,
        predict_with_generate = True,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
    )
    
    test_preds = trainer.predict(test_ds, num_beams=3)
    test_preds = tokenizer.batch_decode(test_preds.predictions, skip_special_tokens=True)
    test_pred_df = pd.DataFrame([[x.strip()] for x in test_preds], columns=['pred'])
    return test_pred_df['pred'].replace('нет', '')
    

def inference(path):
    test_df = pd.read_csv(path)
    tokenizer = GPT2Tokenizer.from_pretrained(CFG.tokenizer_name, local_files_only=True)
    
    def test_prepare_features(examples):
        tokenized_examples = tokenizer(
            ['<LM>' +x for x in examples["name"]],
            padding='max_length', 
            max_length=64,
            truncation=True,
            return_tensors='np'
        )
        return tokenized_examples

    test_ds = Dataset.from_pandas(test_df)
    test_ds = test_ds.map(
        test_prepare_features, batched=True, remove_columns=test_ds.column_names)
    test_df['brand'] = inference_brand(test_ds, tokenizer)
    test_df['good'] = inference_good(test_ds, tokenizer)
    return test_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    df = inference(args.dataset)
    df.to_csv(args.output, index=False)

