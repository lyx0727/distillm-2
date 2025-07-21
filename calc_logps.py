import json
import fire
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_log_probs_for_batch(model, tokenizer, prompts, generations, device):
    texts = [p + g for p, g in zip(prompts, generations)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs[:, :-1, :]
    target_ids = input_ids[:, 1:]

    selected_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    selected_log_probs = selected_log_probs * attention_mask[:, 1:]

    # 根据每个 prompt 长度单独计算每条样本的 generated 部分 logp
    results = []
    from ipdb import set_trace; set_trace()
    for i in range(len(prompts)):
        prompt_ids = tokenizer(prompts[i], return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]
        gen_log_probs = selected_log_probs[i, prompt_len-1 : input_ids.shape[1]-1]
        total_log_prob = gen_log_probs.sum().item()
        results.append(total_log_prob)

    return results

def main(
    model:str,
    data_path:str,
    output_path: str,
    batch_size:int=64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model).to(device)
    model.eval()

    prompts = []
    generations = []
    outputs = []

    with open(data_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in batch
        ]
        generations = [ex["generated_text"] for ex in batch]

        logps = compute_log_probs_for_batch(model, tokenizer, prompts, generations, device)
        for ex, logp in zip(batch, logps):
            ex["log_cumprobs"] = logp
            outputs.append(ex)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)