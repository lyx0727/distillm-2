SEED=200

TEACHER_MODEL_DIR=/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm/Qwen2-7B-Instruct
TEACHER_OUTPUT_DIR=/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/teacher

STUDENT_MODEL_DIR=/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm/Qwen2-1.5B-UltraChat
STUDENT_OUTPUT_DIR=/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/student

OUTPUT_DIR=/AI4M/users/mjzhang/workspace/data/Ultrachat_200k/SPIN_iter1/train

# if  [ ! -d $TEACHER_MODEL_DIR ]; then
#     echo "Teacher model $TEACHER_MODEL_DIR dosen't exist"
#     exit 1
# fi

# if  [ ! -d $STUDENT_MODEL_DIR ]; then
#     echo "Student model $STUDENT_MODEL_DIR dosen't exist"
#     exit 1
# fi

# # 老师学生生成 off-policy 数据

# if [ -d $TEACHER_OUTPUT_DIR ]; then
#     echo "Teacher generation exists, skip."
# else
#     CUDA_VISIBLE_DEVICES=0,1,2,3 python generate/generate_vllm.py \
#         --model $TEACHER_MODEL_DIR \
#         --output_dir $TEACHER_OUTPUT_DIR \
#         --seed $SEED \
#         > instruct_gen_teacher.log 2>&1 &
# fi

# if [ -d $STUDENT_OUTPUT_DIR ]; then
#     echo "Student generation exists, skip."
# else
#     CUDA_VISIBLE_DEVICES=4,5,6,7 python generate/generate_vllm.py \
#         --model $STUDENT_MODEL_DIR \
#         --output_dir $STUDENT_OUTPUT_DIR \
#         --seed $SEED \
#         > instruct_gen_student.log 2>&1 &
# fi

# wait

# # 这里要文件名适配一下
# if [ ! -f $TEACHER_OUTPUT_DIR/train.json ]; then
#     cp $TEACHER_OUTPUT_DIR/output_$SEED.json $TEACHER_OUTPUT_DIR/train.json
# fi
# if [ ! -f $STUDENT_OUTPUT_DIR/train.json ]; then
#     cp $STUDENT_OUTPUT_DIR/output_$SEED.json $STUDENT_OUTPUT_DIR/train.json
# fi

# # 老师学生 off-policy 数据构造 chosen/rejectd 对
# python generate/reformat.py \
#     --teacher_file $TEACHER_OUTPUT_DIR/train.json \
#     --student_file $STUDENT_OUTPUT_DIR/train.json \
#     --output_dir $OUTPUT_DIR

# # 需要适配 tokenizer，会将两个模型重新存到 ckpts 目录下
# python utils/resize_embedding.py \
#     --teacher-model $TEACHER_MODEL_DIR \
#     --student-model $STUDENT_MODEL_DIR \

# 开训
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    --num_processes=4 src/run_distillm.py \
    training_configs/qwen2-1.5b-distillm2.yaml