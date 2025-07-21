
import json, random
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

def calc_acc(item, example, k):
    accs = []
    for answer in item['teacher_answers']:
        if 'accuarcy' in answer:
            accs.append(answer['accuarcy'])
        
    if len(accs) == 0:
        return None
    
    accs = accs[:k]
    return {
        'pass@1': sum(accs) / len(accs),
        f'pass@{len(accs)}': any(acc for acc in accs),
        'difficulty': example['meta']['difficulty'] if example['meta'] else 'TACO',
        '# sample': 1
    }

def calc_adv(item, example, k, reverse=False):
    answers = []
    for answer in item['teacher_answers']:
        if 'accuarcy' in answer:
            answer['score'] = -answer['accuarcy'] / answer['cumulative_logprob']
            answers.append(answer)
        
    if len(answers) < k:
        return None
    
    scores = np.array([answer['score'] for answer in answers])
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        if mean < 1e-5 and not reverse:
            return None
        advs = np.zeros_like(scores)  
    else:
        advs = (scores - mean) / std
    for answer, adv in zip(answers, advs):
        answer['adv'] = adv
        
    answers = sorted(answers, key=lambda x: x['score'], reverse=reverse)

    answers = answers[-k:]
    return answers

def main():
    examples = load_dataset('/AI4M/users/mjzhang/workspace/data/code-r1-12k/data', split='train')
    scores_t = pd.read_json('/AI4M/users/mjzhang/workspace/data/code-r1-12k/Qwen2.5-Coder-7B-Instruct/score.jsonl', lines=True)
    scores_t.drop_duplicates(subset='query', keep='last', inplace=True)
    scores_t['query'] = scores_t['query'].apply(lambda x: json.dumps(x))
    scores_t.set_index('query', inplace=True)
    scores_s = pd.read_json('/AI4M/users/mjzhang/workspace/data/code-r1-12k/qwen2.5-coder-1.5b-sft/score.jsonl', lines=True)
    scores_s['query'] = scores_s['query'].apply(lambda x: json.dumps(x))
    scores_s.set_index('query', inplace=True)
    accs_t, accs_s = [], []
    advs_t, advs_s = [], []
    tok = AutoTokenizer.from_pretrained('/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm/Qwen2.5-Coder-7B-Instruct')
    train_data = []
    dpo_data = []
    for example in tqdm(examples):
        
        messages = example['prompt']
        if messages[0]['role'] == 'system':
            messages = messages[1:]
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        
        item_t = scores_t.loc[json.dumps(example['prompt'])]
        item_s = scores_s.loc[json.dumps(example['prompt'])]
        
        
        accs_t.append(calc_acc(item_t, example, 16))
        accs_s.append(calc_acc(item_s, example, 8))
        
        answers_t = calc_adv(item_t, example, k=4, reverse=False)
        answers_s = calc_adv(item_t, example, k=4, reverse=True)
        
        if answers_s is None or answers_t is None: continue
        
        dpo_data.append(
            {
                'prompt':  messages[0]['content'],
                'chosen':  messages + [{'role': 'assistant', 'content': '```' + answers_t[-1]['teacher_generate_text']}],
                'rejected': messages + [{'role': 'assistant', 'content': '```' + answers_s[-1]['teacher_generate_text']}],
            }
        )
        
        for answer_s in answers_s:
            if answer_s['score'] > 0:
                answer_s['teacher_generate_text'] = 'NULL'
        
        advs_t.extend(item['adv'] for item in answers_t)
        advs_s.extend(item['adv'] for item in answers_s)
        
        
        input_ids = tok.apply_chat_template(messages)
        if len(input_ids) > 2048:
            print(f'[WARN] prompt too long: {len(input_ids)}')
            continue
        for answer_t, answer_s in zip(answers_s, answers_t):
            train_data.append({
                'question': messages,
                'teacher_answer':    answer_t['teacher_generate_text'],
                'student_answer':    answer_s['teacher_generate_text'],
                'teacher_advantage': answer_t['adv'],
                'student_advantage': answer_s['adv'],
            })
    
    
    print("len(train_data)", len(train_data))
    
    with open('/AI4M/users/mjzhang/workspace/Skew-alpha-KL/data/code/train_code-r1-12k-sample-4.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print("item", json.dumps(item, indent=4))
    print("total", len(train_data))
    from sklearn.model_selection import train_test_split
    train_dpo_data, test_dpo_data = train_test_split(dpo_data, test_size=0.01)
    with open('/AI4M/users/mjzhang/workspace/data/code-r1-12k/distillm2/train.json', 'w') as f:
        json.dump(train_dpo_data, f, indent=4)
    with open('/AI4M/users/mjzhang/workspace/data/code-r1-12k/distillm2/test.json', 'w') as f:
        json.dump(test_dpo_data, f, indent=4)
    print("distillm2 item", json.dumps(dpo_data[0], indent=4))
    print("distillm2", len(train_dpo_data))   
    accs_t = [acc for acc in accs_t if acc]
    accs_s = [acc for acc in accs_s if acc]
    accs_t_df = pd.DataFrame(accs_t)
    accs_s_df = pd.DataFrame(accs_s)
    
    print('Teacher: pass@16', accs_t_df['pass@16'].mean())
    print('Student: pass@8', accs_s_df['pass@8'].mean())
    accs_t_df.groupby('difficulty').agg({
        'pass@1': ['mean'],
        'pass@16': ['mean'],
        '# sample': 'sum'
    }).to_json('Qwen2.5-Coder-7B-Instruct_metrics.json', indent=4)
    
    accs_s_df.groupby('difficulty').agg({
        'pass@1': ['mean'],
        'pass@8': ['mean'],
        '# sample': 'sum'
    }).to_json('qwen2.5-coder-1.5b-sft_metrics.json', indent=4)
    
    
    advs_t_df = pd.DataFrame({'teacher': advs_t})
    ax = advs_t_df['teacher'].plot(kind='hist', color='blue', bins=1000, ylim=(0, 500))
    # ax.set_title('Teacher Advantage')
    # ax.figure.savefig('code_teacher_advs.png')
    # ax.cla()
    advs_s_df = pd.DataFrame({'student': advs_s})
    ax = advs_s_df['student'].plot(kind='hist', color='orange', bins=1000, ylim=(0, 500))
    ax.set_title('Code Advantage')
    ax.legend()
    ax.figure.savefig('code_advs.png')
    
if __name__ == '__main__':
    main()