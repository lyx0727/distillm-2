import json, os
import fire
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def calc_win(item):
    win, lose, tie, invalid = 0, 0, 0, 0
    eval = item['judge_response']
    if '</think>' in eval:
        eval = eval[eval.rindex('</think>') + len('</think>'):].strip()
    if '[[A]]' in eval: 
        win += 1
    elif '[[B]]' in eval:
        lose += 1
    elif '[[C]]' in eval: 
        tie += 1
    else: 
        invalid += 1
    
    if item['reverse']:
        tmp = win
        win = lose
        lose = tmp

    return win, lose, tie, invalid

def main(
    data_path: str="/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/train_normalized_logps.json",
    output_path: str="/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/train_win.json",
    mode='teacher',
    ref_path: str = "/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/Llama-3-8B-Instruct/train.json",
    judge_model: str = "/AI4M/users/mjzhang/llm/Qwen3-32B",
    batch_size: int = 4096,
    judge_generation_path: str = "/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/judge-Qwen3-32B.jsonl",
):
    system = [
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below.",
        "You should choose the assistant that follows the user’s instructions and answers the user’s question better.",
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.", 
        "Begin your evaluation by comparing the two responses and provide a short explanation.", 
        "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.", 
        "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants.", 
        """Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.""",
    ]
    print("data_path", data_path.format(mode=mode))
    with open(data_path.format(mode=mode)) as f:
        data = json.load(f)
    with open(ref_path) as f:
        data_ref = json.load(f)
        
    judge_generation_path = judge_generation_path.format(mode=mode)
    print("judge_generation_path", judge_generation_path)
    output_path = output_path.format(mode=mode)
    print("output_path", output_path)
        
    import random
    print(json.dumps(random.choice(data), indent=4))

    data = {
        item['prompt']: item
        for item in data
    }
    data_ref = {
        item['prompt']: item
        for item in data_ref
    }

    samples = []
    for reverse in [True, False]:
        for prompt, item_a in data.items():
            answer_a = item_a['generated_text']
            answer_b = data_ref[prompt]['generated_text']
            if reverse:
                answer_tmp = answer_b
                answer_b = answer_a
                answer_a = answer_tmp

            judge_prompt = f"""[System]\n{' '.join(system)}\n\n[User Question]\n{prompt}\n\n[The Start of Assistant A’s Answer]\n{answer_a}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer_b}\n[The End of Assistant B’s Answer]"""
        
            samples.append((judge_prompt, prompt, reverse))

    
    print("samples", len(samples))    


    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    prompts = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': sample[0]}], 
            tokenize=False, 
            add_generation_prompt=True,
        )
        for sample in samples 
    ]
    model = None
    if os.path.exists(judge_generation_path):
        with open(judge_generation_path) as f:
            judge_results = [json.loads(line) for line in f]
    else:
        judge_results = []
    for i in tqdm(range(0, len(prompts[len(judge_results):]), batch_size), desc="Generating batch..."):  
        batch_prompts = prompts[i : i + batch_size]
        if model is None:
            model = LLM(judge_model, tensor_parallel_size=torch.cuda.device_count())
        batch_generations = model.generate(batch_prompts, SamplingParams(n=1, max_tokens=4096))
        batch_samples = samples[i : i + batch_size]
        for judge_prompt, judge_gen, (_, prompt, reverse) in zip(batch_prompts, batch_generations, batch_samples):
            result = {
                    'judge_prompt': judge_prompt,
                    'judge_response': judge_gen.outputs[0].text,
                    'prompt': prompt,
                    'answer_a': data[prompt],
                    'answer_b': data_ref[prompt],
                    'reverse': reverse
                }
            with open(judge_generation_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
            judge_results.append(result)
            
    all_win, all_lose, all_tie, all_invalid = 0, 0, 0, 0
    all_win_rev, all_lose_rev, all_tie_rev, all_invalid_rev = 0, 0, 0, 0
    judge_results_map = {}
    for item in tqdm(judge_results, desc="Scoring"):
        win, lose, tie, invalid = calc_win(item)
        # item['score'] = 1.0 if win else (0.5 if tie else 0.)
        item['score'] = 1 if win else 0
        
        judge_results_map.setdefault(item['prompt'], []).append(item)
        if item['reverse']:
            all_win_rev += win
            all_lose_rev += lose
            all_tie_rev += tie
            all_invalid_rev += invalid
        else:
            all_win += win
            all_lose += lose
            all_tie += tie
            all_invalid += invalid
    total = len(judge_results) // 2
    assert total == all_win + all_lose + all_tie + all_invalid
    print("total:", total)
    print("win: ", all_win / total)
    print("lose: ", all_lose / total)
    print("tie: ", all_tie / total)
    print("invalid: ", all_invalid / total)
    
    print("win(rev): ", all_win_rev / total)
    print("lose(rev): ", all_lose_rev / total)
    print("tie(rev): ", all_tie_rev / total)
    print("invalid(rev): ", all_invalid_rev / total)
    
    outputs = []
    for prompt, items in judge_results_map.items():
        assert len(items) == 2
        assert (items[0]['reverse'] and not items[1]['reverse']) or (items[1]['reverse'] and not items[0]['reverse'])
        assert items[0]['answer_a'] == items[1]['answer_a']
        assert items[0]['answer_b'] == items[1]['answer_b']
        
        item = data[prompt] | {
            'score': (items[0]['score'] + items[1]['score']) / 2
        }
        outputs.append(item)
        
    outputs = pd.DataFrame(outputs)
    outputs['normalized_score'] = outputs.apply(lambda x: x['score'] / (-x['normalized_log_cumprobs']) if x['score'] > 0 else -1, axis=1)
    outputs['adv'] = (outputs['normalized_score'] - outputs['normalized_score'].mean()) / outputs['normalized_score'].std()
    
    outputs.to_json(output_path, orient='records', indent=4, force_ascii=False)
    
    ax = outputs['score'].plot(kind='hist')
    ax.set_title(mode.capitalize() + ' Win Score')
    ax.figure.savefig(output_path.replace('.json', '_score.png'))
    
    ax.cla()
    
    ax = outputs['adv'].plot(kind='hist', bins=10000, xlim=[-1.0, 1.0])
    ax.set_title(mode.capitalize() + ' Advantage')
    ax.figure.savefig(output_path.replace('.json', '_adv.png'))
    
    
if __name__ == "__main__":
    fire.Fire(main)