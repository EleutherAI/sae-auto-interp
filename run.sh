sleep 3060
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server --model=casperhansen/llama-3-70b-instruct-awq --quantization=awq --disable-log-requests --enable-prefix-caching --max-num-seqs 1000 --gpu-memory-utilization=0.9 --port=8002 --tensor-parallel-size=8
ython extras/scripts/explain_gpt2many.py --layers 0,4,8