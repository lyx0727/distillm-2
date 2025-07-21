import json, os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from rouge_metric import compute_metrics

def handle_score(rank, mode, output_path, examples, data: np.ndarray):
    
    gold_answer_map = {
        example[0]['content']: example[1]['content']
        for example in examples
    }
    
    for item in tqdm(data, position=rank, desc=f'Scoring {rank}: '):
        gold_answer = gold_answer_map[item['prompt']]
        answer = item['generated_text']
        item['score'] = compute_metrics([answer], [[gold_answer]])['rougeL']
        with open(output_path.format(mode=mode).replace('json', f'_{rank}.jsonl'), 'a') as f:
            f.write(json.dumps(item) + '\n')
    with open(output_path.format(mode=mode).replace('json', f'_{rank}.json'), 'w') as f:
        json.dump(data.tolist(), f, indent=4, ensure_ascii=False)

def main(
    data_path: str="/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/train.json",
    output_path: str="/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/train_rouge.json",
    mode='teacher',
):
    examples = load_dataset(f'/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1', split='train')['real']

    data_path = data_path.format(mode=mode)
    with open(data_path) as f:
        data = json.load(f)
    # breakpoint()

    num_threads = 4
    data_list = np.array_split(data, num_threads)
    threads = []
    for i in range(num_threads):
        if os.path.exists(output_path.format(mode=mode).replace('json', f'_{i}.json')):
            continue 
        thread = mp.Process(target=handle_score, args=(i, mode, output_path, examples, data_list[i]))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
        
    all_results = []
    for i in range(num_threads):
        with open(output_path.format(mode=mode).replace('json', f'_{i}.json')) as f:
            results = json.load(f)
        all_results += results   
    with open(output_path.format(mode=mode), 'w') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)   

if __name__ == '__main__':
    import fire
    fire.Fire(main)