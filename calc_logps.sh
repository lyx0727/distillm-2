TEACHER_MODEL_DIR=/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm/Qwen2-7B-Instruct
TEACHER_OUTPUT_DIR=/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/teacher

STUDENT_MODEL_DIR=/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm/Qwen2-1.5B-UltraChat
STUDENT_OUTPUT_DIR=/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/student

# CUDA_VISIBLE_DEVICES=7 python calc_logps.py \
#     --model $STUDENT_MODEL_DIR \
#     --data_path $STUDENT_OUTPUT_DIR/train_win.json \
#     --output_path $STUDENT_OUTPUT_DIR/train_logps.json \
#     --batch_size 16 \
#     > calc_logps_student.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python calc_logps.py \
#     --model $TEACHER_MODEL_DIR \
#     --data_path $TEACHER_OUTPUT_DIR/train_win.json \
#     --output_path $TEACHER_OUTPUT_DIR/train_logps.json \
#     --batch_size 8 \
#     > calc_logps_teacher.log 2>&1 &

python normalize_logps.py \
    --model $STUDENT_MODEL_DIR \
    --data_path $STUDENT_OUTPUT_DIR/train_logps.json \
    --output_path $STUDENT_OUTPUT_DIR/train_normalized_logps.json

python normalize_logps.py \
    --model $TEACHER_MODEL_DIR \
    --data_path $TEACHER_OUTPUT_DIR/train_logps.json \
    --output_path $TEACHER_OUTPUT_DIR/train_normalized_logps.json