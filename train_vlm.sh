MODEL_PATH_ORG=/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm
MODEL_PATH=/AI4M/users/mjzhang/workspace/distillm-2/ckpts
TEACHER_MODEL_NAME=llava-1.5-7b-hf
STUDENT_MODEL_NAME=tiny-llava-v1-hf
DATA_DIR=/AI4M/users/mjzhang/workspace/data/textvqa
SEED=42

# 需要适配 tokenizer，会将两个模型重新存到 ckpts 目录下
# python utils/resize_embedding_vlm.py \
    # --teacher-model $MODEL_PATH_ORG/$TEACHER_MODEL_NAME \
    # --student-model $MODEL_PATH_ORG/$STUDENT_MODEL_NAME \

if [ ! -d $DATA_DIR/teacher ]; then
    echo "Generating teacher..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 python generate/generate_vllm_vlm.py \
        --data_dir $DATA_DIR/data \
        --model  $MODEL_PATH/$TEACHER_MODEL_NAME \
        --tokenizer $MODEL_PATH/$TEACHER_MODEL_NAME \
        --output_dir $DATA_DIR/teacher \
        --split train \
        --seed $SEED \
        > vlm_gen_teacher.log 2>&1 &
fi

if [ ! -d $DATA_DIR/student ]; then
    echo "Generating student..."
    CUDA_VISIBLE_DEVICES=4,5,6,7 python generate/generate_vllm_vlm.py \
        --data_dir $DATA_DIR/data \
        --model $MODEL_PATH/$STUDENT_MODEL_NAME \
        --tokenizer $MODEL_PATH/$STUDENT_MODEL_NAME \
        --output_dir $DATA_DIR/student \
        --split train \
        --seed $SEED \
        > vlm_gen_student.log 2>&1 &
fi

wait

# 这里要文件名适配一下
if [ ! -f $DATA_DIR/teacher/train.json ]; then
    cp $DATA_DIR/teacher/output_$SEED.json $DATA_DIR/teacher/train.json
fi
if [ ! -f $DATA_DIR/student/train.json ]; then
    cp $DATA_DIR/student/output_$SEED.json $DATA_DIR/student/train.json
fi

# # 老师学生 off-policy 数据构造 chosen/rejectd 对
python generate/reformat.py \
    --teacher_file $DATA_DIR/teacher/train.json \
    --student_file $DATA_DIR/student/train.json \
    --output_dir $DATA_DIR/distillm2

# # 开训
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file accelerate_configs/deepspeed_zero0.yaml \
    --num_processes=8 \
    --main_process_port=29888 \
    src/run_distivlm.py \
    training_configs/vlm.yaml
