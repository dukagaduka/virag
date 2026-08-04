#!/usr/bin/env bash
set -euo pipefail

VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-embeddings-model}"
VLLM_API_KEY="${VLLM_API_KEY:-}"

VLLM_RUNNER="${VLLM_RUNNER:-pooling}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.22}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"

VLLM_TOKENIZER_MODE="${VLLM_TOKENIZER_MODE:-auto}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-auto}"
VLLM_POOLER_CONFIG="${VLLM_POOLER_CONFIG:-}"
VLLM_UVICORN_LOG_LEVEL="${VLLM_UVICORN_LOG_LEVEL:-warning}"

ARGS=(
  "/model"
  "--host" "0.0.0.0"
  "--port" "8000"
  "--served-model-name" "$VLLM_SERVED_MODEL_NAME"
  "--runner" "$VLLM_RUNNER"
  "--tensor-parallel-size" "$VLLM_TENSOR_PARALLEL_SIZE"
  "--pipeline-parallel-size" "$VLLM_PIPELINE_PARALLEL_SIZE"
  "--max-model-len" "$VLLM_MAX_MODEL_LEN"
  "--gpu-memory-utilization" "$VLLM_GPU_MEMORY_UTILIZATION"
  "--max-num-seqs" "$VLLM_MAX_NUM_SEQS"
  "--max-num-batched-tokens" "$VLLM_MAX_NUM_BATCHED_TOKENS"
  "--tokenizer-mode" "$VLLM_TOKENIZER_MODE"
  "--dtype" "$VLLM_DTYPE"
  "--load-format" "$VLLM_LOAD_FORMAT"
  "--uvicorn-log-level" "$VLLM_UVICORN_LOG_LEVEL"
  "--generation-config" "vllm"
  "--enable-request-id-headers"
)

if [[ -n "$VLLM_API_KEY" ]]; then
  ARGS+=("--api-key" "$VLLM_API_KEY")
fi

if [[ "$VLLM_TRUST_REMOTE_CODE" == "1" ]]; then
  ARGS+=("--trust-remote-code")
fi

if [[ -n "$VLLM_POOLER_CONFIG" ]]; then
  ARGS+=("--pooler-config" "$VLLM_POOLER_CONFIG")
fi

echo "vLLM embeddings model: $MODEL_NAME"
echo "served model name:     $VLLM_SERVED_MODEL_NAME"
echo "listen:                $HOST:$PORT"
echo "metrics:               /metrics"
echo "OpenAI endpoint:       /v1/embeddings"

exec vllm serve "${ARGS[@]}"
