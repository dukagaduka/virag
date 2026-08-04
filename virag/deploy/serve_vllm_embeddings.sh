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

# LoRA adapter (optional). The base model is always mounted at /model; a LoRA
# adapter, when used, is mounted at the fixed path /adapter (only the host path
# varies, via VLLM_DEPLOY_ADAPTER_PATH in compose). When an adapter is present the
# base is served under VLLM_BASE_MODEL_NAME and the adapter is registered under
# VLLM_SERVED_MODEL_NAME — so clients (virag) query VLLM_SERVED_MODEL_NAME and
# transparently get base+adapter.
VLLM_BASE_MODEL_NAME="${VLLM_BASE_MODEL_NAME:-base-model}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
ADAPTER_DIR="/adapter"

# Optional weight quantization (e.g. fp8) and eager mode — mainly for fitting a
# large model on a small GPU. Both are off by default.
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

if [[ -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  LORA_ENABLED=1
  SERVED_MODEL_NAME="$VLLM_BASE_MODEL_NAME"
else
  LORA_ENABLED=0
  SERVED_MODEL_NAME="$VLLM_SERVED_MODEL_NAME"
fi

ARGS=(
  "/model"
  "--host" "0.0.0.0"
  "--port" "8000"
  "--served-model-name" "$SERVED_MODEL_NAME"
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

if [[ "$LORA_ENABLED" == "1" ]]; then
  ARGS+=(
    "--enable-lora"
    "--max-lora-rank" "$VLLM_MAX_LORA_RANK"
    "--lora-modules" "${VLLM_SERVED_MODEL_NAME}=${ADAPTER_DIR}"
  )
fi

if [[ -n "$VLLM_QUANTIZATION" ]]; then
  ARGS+=("--quantization" "$VLLM_QUANTIZATION")
fi

if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  ARGS+=("--enforce-eager")
fi

echo "vLLM base model dir:   /model"
if [[ "$LORA_ENABLED" == "1" ]]; then
  echo "LoRA adapter:          $ADAPTER_DIR (max_lora_rank=$VLLM_MAX_LORA_RANK)"
  echo "base served name:      $SERVED_MODEL_NAME"
  echo "client model name:     $VLLM_SERVED_MODEL_NAME (base+adapter)"
else
  echo "served model name:     $SERVED_MODEL_NAME"
fi
echo "listen:                0.0.0.0:8000"
echo "metrics:               /metrics"
echo "OpenAI endpoint:       /v1/embeddings"

exec vllm serve "${ARGS[@]}"
