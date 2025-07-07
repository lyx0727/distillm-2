tasks=("alpaca-eval" "evol-instruct" "ultrafeedback")
devices=("$@")
seed=200

echo "eval instruct: $MODEL"

for i in "${!tasks[@]}"; do
    if [[ "$i" -lt "$#" && "${devices[$i]}" -ge 0 ]]; then
        echo "Generate ${tasks[$i]} on cuda:${devices[$i]}"
        
        OUTPUT_DIR=results/${tasks[$i]}/${MODEL}
        if [[ -f "$OUTPUT_DIR/output_$seed.json" ]]; then
            continue
        fi
        mkdir -p $OUTPUT_DIR
        set -x
        CUDA_VISIBLE_DEVICES=${devices[$i]} python generate/generate_vllm.py \
            --model $MODEL_PATH \
            --output_dir ${OUTPUT_DIR} \
            --data_dir ${tasks[$i]} \
            --seed $seed \
        > ${OUTPUT_DIR}/generate.log 2>&1 &
        set +x
    fi
done

wait

ref_paths=("eval/alpacaeval/alpaca_eval.json" "eval/evol-instruct/evol_inst_eval.json" "eval/ultrafeedback/ultrafeedback_eval.json")
for i in "${!tasks[@]}"; do
    if [[ "$i" -lt "$#" && "${devices[$i]}" -ge 0 ]]; then
        echo "Judge ${tasks[$i]}"
        OUTPUT_DIR=results/${tasks[$i]}/$MODEL
        EXP_NAME=$MODEL
        python eval/build_evaluation.py \
            --data-path1 $OUTPUT_DIR/output_$seed.json \
            --data-path2 ${ref_paths[$i]} \
            --pairwise \
            --output-file ${EXP_NAME}-${tasks[$i]}_$seed \
            --judge gpt-4o

        python eval/build_evaluation.py \
            --data-path2 $OUTPUT_DIR/output_$seed.json \
            --data-path1 ${ref_paths[$i]} \
            --pairwise \
            --output-file ${tasks[$i]}-${EXP_NAME}_$seed \
            --judge gpt-4o

        bash eval/run.sh $EXP_NAME ${tasks[$i]} $OUTPUT_DIR
    fi
done
