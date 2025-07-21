CUDA_VISIBLE_DEVICES=0,1 python score_win_rate.py \
    --mode teacher \
    # > instruct_score_teacher.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 python score_win_rate.py \
    --mode student \
    # > instruct_score_student.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1,2,3 python score_win_rate.py \
#     --mode=teacher \
#     --ref_path="/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/student/train.json" \
#     --output_path="/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/train_test.json" \
#     --judge_generation_path /AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/{mode}/judge-Qwen3-32B_test.jsonl \
#     > instruct_score_teacher_vs_student.log 2>&1 &