import json
import fire

def main(
    data_path: str = '/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/train/train.json',
    teacher_path: str = '/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/teacher/train_win.json',
    student_path: str = '/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/student/train_win.json',
    output_path: str = '/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/train.json',
):
    with open(data_path) as f:
        data = json.load(f)
    with open(teacher_path) as f:
        teacher_data = json.load(f)
    with open(student_path) as f:
        student_data = json.load(f)

    teacher_data = {
        item['prompt']: item
        for item in teacher_data
    }
    student_data = {
        item['prompt']: item
        for item in student_data
    }
    outputs = []
    for item in data:
        prompt = item['prompt']
        item_t = teacher_data[prompt]
        item_s = student_data[prompt]
        
        new_item = {
            'prompt': prompt,
            'chosen': [{'content': prompt, 'role': 'user'}, {'content': item_t['generated_text'], 'role': 'assistant'}],
            'rejected': [{'content': prompt, 'role': 'user'}, {'content': item_s['generated_text'], 'role': 'assistant'}],
            'chosen_adv': item_t['adv'],
            'rejected_adv': item_s['adv']
        }
        for key in ['chosen', 'rejected']:
            assert json.dumps(item[key]) == json.dumps(new_item[key]), breakpoint()
        
        outputs.append(new_item)
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=4)
        
if __name__ == "__main__":
    fire.Fire(main)