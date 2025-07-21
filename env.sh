export OPENAI_BASE_URL=https://api2.aigcbest.top/v1
export OPENAI_API_KEY=sk-zBNMSTsi1NfsxmsGfzgxI13Ja2QccfXJmzSWoDicfGdQ2B1k

# export MODEL=Qwen2-1.5B-Distillm2
# export MODEL_PATH=/AI4M/users/mjzhang/workspace/distillm-2/outputs/qwen2-1.5b-distillm2/merged

# export MODEL=Qwen2-1.5B-Distillm2-ours-epoch-3
# export MODEL_PATH=/AI4M/users/mjzhang/workspace/distillm-2-ours/outputs/qwen2-1.5b-distillm2-epoch-3/merged

export MODEL=Qwen2.5-Coder-1.5B-Distillm2
export MODEL_PATH=/AI4M/users/mjzhang/workspace/distillm-2/outputs/qwen2.5-coder-1.5b-distillm2

if [ ! -d $MODEL_PATH/merged ]; then
    echo "Merge lora..."
    python utils/merging.py \
        --base-model-name /AI4M/users/mjzhang/workspace/distillm-2/ckpts/qwen2.5-coder-1.5b-sft \
        --lora-model-name $MODEL_PATH
fi

export MODEL_PATH=$MODEL_PATH/merged