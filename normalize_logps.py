import json, fire
from tqdm import tqdm
from transformers import AutoTokenizer


def main(
    model:str,
    data_path:str,
    output_path: str,
): 
    tokenizer = AutoTokenizer.from_pretrained(model)
    with open(data_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
        
    for item in tqdm(data):
        gen_tokens = tokenizer.encode(item['generated_text'], add_special_tokens=False)
        item['normalized_log_cumprobs'] = item['log_cumprobs'] / len(gen_tokens)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        

if __name__ == '__main__':
    fire.Fire(main)