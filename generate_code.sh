export PATH=/AI4M/users/mjzhang/workspace/firejail/bin:$PATH

DATA_PATH=/AI4M/users/mjzhang/workspace/data/code-r1-12k
SEED=200

# TEACHER_MODEL=deepseek-coder-6.7b-instruct
# STUDENT_MODEL=deepseek-coder-1.3b-sft
TEACHER_MODEL=Qwen2.5-Coder-7B-Instruct
STUDENT_MODEL=qwen2.5-coder-1.5b-sft


STUDENT_MODEL_DIR="/AI4M/users/mjzhang/workspace/distillm-2/outputs/$STUDENT_MODEL"
STUDENT_OUTPUT_DIR=$DATA_PATH/$STUDENT_MODEL

if [ ! -d $STUDENT_MODEL_DIR ]; then
    echo "Student model doesn't exists"
    exit 1
fi

if [ ! -d $STUDENT_OUTPUT_DIR ]; then
    mkdir -p $STUDENT_OUTPUT_DIR
fi

if [ -f $STUDENT_OUTPUT_DIR/train.json ]; then
    echo "$STUDENT_OUTPUT_DIR/train.json exists, skip"
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 python /AI4M/users/mjzhang/workspace/Skew-alpha-KL/data_construct/data_construct_code_off_policy.py \
            --dataset_path $DATA_PATH/data \
            --model_name_or_path $STUDENT_MODEL_DIR \
            --output_path $STUDENT_OUTPUT_DIR/train.json \
            --raw_output_path $STUDENT_OUTPUT_DIR/raw.json \
            --raw_score_path $STUDENT_OUTPUT_DIR/score.jsonl \
            --seed $SEED \
            > code_gen_student_$STUDENT_MODEL.log 2>&1 &
fi

TEACHER_MODEL_DIR="/AI4M/users/mjzhang/workspace/Skew-alpha-KL/llm/$TEACHER_MODEL"
TEACHER_OUTPUT_DIR=$DATA_PATH/$TEACHER_MODEL

if [ ! -d $TEACHER_MODEL_DIR ]; then
    echo "Teacher model doesn't exists"
    exit 1
fi

if [ ! -d $TEACHER_OUTPUT_DIR ]; then
    mkdir -p $TEACHER_OUTPUT_DIR
fi

if [ -f $TEACHER_OUTPUT_DIR/train.json ]; then
    echo "$TEACHER_OUTPUT_DIR/train.json exists, skip"
else
    CUDA_VISIBLE_DEVICES=4,5,6,7 python /AI4M/users/mjzhang/workspace/Skew-alpha-KL/data_construct/data_construct_code_off_policy.py \
            --dataset_path $DATA_PATH/data \
            --model_name_or_path $TEACHER_MODEL_DIR \
            --output_path $TEACHER_OUTPUT_DIR/train.json \
            --raw_output_path $TEACHER_OUTPUT_DIR/raw.json \
            --raw_score_path $TEACHER_OUTPUT_DIR/score.jsonl \
            --seed $SEED \
            --gpu_memory_utilization 0.9 \
            > code_gen_teacher_$TEACHER_MODEL.log 2>&1 &
fi