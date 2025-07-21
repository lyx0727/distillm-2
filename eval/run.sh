seed=200
output1=results/outputs/${1}-${2}_$seed.jsonl
output2=results/outputs/${2}-${1}_$seed.jsonl

if [[ -f "$output1" ]]; then
  echo "Skipping ${1}-${2}"
else
  python eval/api_request_parallel_processor.py \
  --requests_filepath results/inputs/${1}-${2}_$seed.jsonl \
  --save_filepath $output1 \
  --request_url $OPENAI_BASE_URL/chat/completions \
  --max_requests_per_minute 500 \
  --max_tokens_per_minute 625000 \
  --max_attempts 5 \
  --logging_level 20 \
  --api_key $OPENAI_API_KEY
fi

if [[ -f "$output2" ]]; then
  echo "Skipping ${2}-${1}"
else
  python eval/api_request_parallel_processor.py \
    --requests_filepath results/inputs/${2}-${1}_$seed.jsonl \
    --save_filepath $output2 \
    --request_url $OPENAI_BASE_URL/chat/completions \
    --max_requests_per_minute 500 \
    --max_tokens_per_minute 625000 \
    --max_attempts 5 \
    --logging_level 20 \
    --api_key $OPENAI_API_KEY
fi

python eval/grading.py \
  --input1 $output1 \
  --input2 $output2 \
  --output_dir $3 \
  --pairwise