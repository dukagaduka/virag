#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# serve_reranker.sh
#
# Назначение:
#   Запуск vLLM endpoint для мультимодального реранкера Qwen3-VL-Reranker
#   (/v1/rerank, /score) с:
#   - pooling runner'ом и hf_overrides, превращающими модель в классификатор yes/no
#   - обязательным chat template (без него скоры мусорные, а ошибки нет)
#   - ограничением картинок и их разрешения (max_pixels)
#   - опциональной квантизацией весов / eager режимом для маленьких GPU
#
# Все параметры тюнятся прямо здесь, а не через .env: .env содержит только
# deploy-настройки (образ, порт, пути), см. docker-compose.yaml.
#
# Требуется vLLM >= 0.14.0.
###############################################################################

############################################
# REQUIRED
############################################
MODEL_NAME="/model"
# Путь к директории модели внутри контейнера.
# Compose монтирует туда RERANKER_DEPLOY_MODEL_PATH, обычно менять не нужно.

CHAT_TEMPLATE="/chat_template.jinja"
# Jinja-шаблон промпта «Query / Document -> yes/no».
# Compose монтирует туда qwen3_vl_reranker.jinja из этой папки.
# Без него сервер стартует, но скоры будут бессмысленными.
# Инструкцию для задачи меняй в default(...) прямо в шаблоне: через /v1/rerank
# она отдельным полем не передаётся.

############################################
# LISTEN / AUTH
############################################
HOST="0.0.0.0"
# Адрес, на котором слушает API.
# 0.0.0.0 — слушать на всех интерфейсах контейнера.
# 127.0.0.1 — только локально внутри контейнера.

PORT="8000"
# Порт OpenAI-compatible API внутри контейнера.
# Наружу он пробрасывается через RERANKER_DEPLOY_PORT в .env, здесь менять не нужно.

API_KEY="${VLLM_API_KEY:-}"
# Bearer token, который сервер требует на /v1/*; пусто — без авторизации.
# Задаётся в .env (RERANKER_DEPLOY_API_KEY): compose передаёт его в контейнер как
# VLLM_API_KEY, им же ходит healthcheck. Здесь не меняй.

SERVED_MODEL_NAME="qwen3-vl-reranker"
# Имя модели, которое клиент указывает в поле "model".

############################################
# RUNNER
############################################
RUNNER="pooling"
# Режим работы модели.
# pooling — модель работает как классификатор, а не как генератор.
# Для реранкера менять нельзя.

HF_OVERRIDES='{"architectures": ["Qwen3VLForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
# Подмена конфига модели: голова классификатора собирается из строк "no"/"yes"
# lm_head, score = sigmoid(logit(yes) - logit(no)).
# Это параметры именно оригинального Qwen3-VL-Reranker; для другой модели
# меняй осознанно.

############################################
# PARALLELISM
############################################
TENSOR_PARALLEL_SIZE="1"
# На сколько GPU распиливается одна модель по тензорному параллелизму.
# Для реранкера 2B–8B обычно хватает одной GPU.

PIPELINE_PARALLEL_SIZE="1"
# Пайплайновый параллелизм.
# Для обычных запусков оставляй 1.

############################################
# MODEL / MEMORY
############################################
MAX_MODEL_LEN="8192"
# Максимальная длина пары query + document (в токенах).
# Скриншот даёт сотни-тысячи image-токенов; если пара не влезает, запрос падает.
# Ограничивай не только это значение, но и MM_MAX_PIXELS.
# Уменьшай, если сервер не стартует или слишком жрёт VRAM.

GPU_MEMORY_UTILIZATION="0.9"
# Целевая доля VRAM, которую vLLM старается занять.
# Если GPU делится с другими сервисами (например, с retriever) — снижай.
# Слишком высокое значение может привести к OOM или нестабильному старту.

MAX_NUM_SEQS="8"
# Максимальное число пар, обрабатываемых одновременно.
# Один запрос с 20 кандидатами — это 20 пар.
# Больше — выше пропускная способность, но выше пик памяти.
# Меняй вместе с MAX_NUM_BATCHED_TOKENS, а не изолированно.

MAX_NUM_BATCHED_TOKENS="8192"
# Потолок суммарных токенов в одном scheduler batch.
# Больше — лучше утилизация GPU, но выше пики памяти.
# Для pooling-моделей без chunked prefill не должен быть меньше MAX_MODEL_LEN.

############################################
# MULTIMODAL
############################################
LIMIT_MM_PER_PROMPT='{"image": 1}'
# Максимум картинок на одну пару.
# 1 — картинка только в документе (запрос текстовый).
# 2 — если запрос тоже может быть картинкой.

MM_MAX_PIXELS="1048576"
# Максимум пикселей на картинку (ширина * высота), остальное уменьшается.
# Главный рычаг баланса качества и скорости: у Qwen3-VL один image-токен
# соответствует ~32x32 пикселям, то есть 1048576 ≈ 1024 токена.
# Подбирай так же, как у ретривера, чтобы реранкер видел страницу не хуже него.
# Визуальный энкодер на высоком разрешении — основная статья затрат.

ALLOWED_LOCAL_MEDIA_PATH="/media"
# Директория, из которой разрешено брать картинки по file:///media/... вместо base64.
# Compose монтирует туда RERANKER_DEPLOY_MEDIA_PATH.
# Пусто — запретить file:// (тогда картинки только как data:/http URL).

############################################
# DTYPE / EXECUTION / QUANTIZATION
############################################
DTYPE="auto"
# Тип вычислений.
# auto — дать vLLM выбрать подходящий dtype.
# Можно задать явно: bfloat16 / float16.

QUANTIZATION=""
# Квантизация весов на лету.
# Пусто — полная точность.
# fp8 — вдвое меньше памяти под веса (нужна Ada/Hopper).
# Перед использованием проверь, что качество реранка не просело.

ENFORCE_EAGER="0"
# 1 — не захватывать CUDA graphs, экономит VRAM ценой скорости.
# 0 — обычный режим.

TRUST_REMOTE_CODE="0"
# Разрешение выполнять custom code из model repository.
# 0 — безопаснее, но некоторые модели не заведутся без этого.
# 1 — включай только если модель действительно этого требует и источник доверенный.

############################################
# OBSERVABILITY / LOGGING
############################################
UVICORN_LOG_LEVEL="warning"
# Уровень логирования frontend-сервера.
# error — минимум шума
# warning — обычно оптимально
# info — больше диагностической информации
# debug — для глубокой отладки, может быть очень шумно

############################################
# EXTRA / EXPERIMENTAL
############################################
EXTRA_ARGS=()
# Сюда складываются дополнительные флаги, которые не хочется выносить
# в отдельные переменные.

EXTRA_ARGS+=("--generation-config" "vllm")
# Использовать generation config от vLLM.
# Это делает поведение более предсказуемым со стороны сервера.
# Меняй только если осознанно хочешь поведение из model repo.

############################################
# Build argv
############################################
ARGS=()

# Core
ARGS+=("$MODEL_NAME")
ARGS+=("--host" "$HOST")
ARGS+=("--port" "$PORT")
ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")

# Auth
if [[ -n "$API_KEY" ]]; then
  ARGS+=("--api-key" "$API_KEY")
fi

# Runner
ARGS+=("--runner" "$RUNNER")
ARGS+=("--hf-overrides" "$HF_OVERRIDES")
ARGS+=("--chat-template" "$CHAT_TEMPLATE")

# Parallelism
ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
ARGS+=("--pipeline-parallel-size" "$PIPELINE_PARALLEL_SIZE")

# Model / memory
ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")

# Multimodal
ARGS+=("--limit-mm-per-prompt" "$LIMIT_MM_PER_PROMPT")
ARGS+=("--mm-processor-kwargs" "{\"max_pixels\": $MM_MAX_PIXELS}")

if [[ -n "$ALLOWED_LOCAL_MEDIA_PATH" ]]; then
  ARGS+=("--allowed-local-media-path" "$ALLOWED_LOCAL_MEDIA_PATH")
fi

# Dtype / execution
ARGS+=("--dtype" "$DTYPE")

if [[ -n "$QUANTIZATION" ]]; then
  ARGS+=("--quantization" "$QUANTIZATION")
fi

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  ARGS+=("--enforce-eager")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  ARGS+=("--trust-remote-code")
fi

# Logging
ARGS+=("--enable-request-id-headers")
ARGS+=("--uvicorn-log-level" "$UVICORN_LOG_LEVEL")

# Extra
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

############################################
# Pretty print config
############################################
echo "vLLM model:              $MODEL_NAME"
echo "listen:                  $HOST:$PORT"
echo "api key:                 $([[ -n "$API_KEY" ]] && echo enabled || echo disabled)"
echo "served model name:       $SERVED_MODEL_NAME"
echo "runner:                  $RUNNER"
echo "chat template:           $CHAT_TEMPLATE"
echo "hf overrides:            $HF_OVERRIDES"
echo
echo "tensor parallel:         $TENSOR_PARALLEL_SIZE"
echo "pipeline parallel:       $PIPELINE_PARALLEL_SIZE"
echo
echo "max model len:           $MAX_MODEL_LEN"
echo "gpu mem utilization:     $GPU_MEMORY_UTILIZATION"
echo "max num seqs:            $MAX_NUM_SEQS"
echo "max batched tokens:      $MAX_NUM_BATCHED_TOKENS"
echo
echo "images per prompt:       $LIMIT_MM_PER_PROMPT"
echo "max pixels:              $MM_MAX_PIXELS"
echo "local media path:        $([[ -n "$ALLOWED_LOCAL_MEDIA_PATH" ]] && echo "$ALLOWED_LOCAL_MEDIA_PATH" || echo disabled)"
echo
echo "dtype:                   $DTYPE"
echo "quantization:            $([[ -n "$QUANTIZATION" ]] && echo "$QUANTIZATION" || echo disabled)"
echo "enforce eager:           $([[ "$ENFORCE_EAGER" == "1" ]] && echo enabled || echo disabled)"
echo "trust remote code:       $([[ "$TRUST_REMOTE_CODE" == "1" ]] && echo enabled || echo disabled)"
echo
echo "uvicorn log level:       $UVICORN_LOG_LEVEL"
echo "OpenAI endpoint:         /v1/rerank (also /score)"
echo "metrics:                 /metrics"
echo
echo "full command:"
printf ' %q' vllm serve "${ARGS[@]}"
echo
echo

exec vllm serve "${ARGS[@]}"
