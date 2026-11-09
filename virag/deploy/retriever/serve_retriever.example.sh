#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# serve_retriever.sh
#
# Назначение:
#   Запуск одного OpenAI-compatible vLLM endpoint для эмбеддингов (/v1/embeddings) с:
#   - pooling runner'ом
#   - опциональным LoRA adapter'ом (подхватывается автоматически, если смонтирован)
#   - опциональной квантизацией весов / eager режимом для маленьких GPU
#   - базовыми параметрами памяти/батчинга
#
# Все параметры тюнятся прямо здесь, а не через .env: .env содержит только
# deploy-настройки (образ, порт, пути к модели/адаптеру), см. docker-compose.yaml.
###############################################################################

############################################
# REQUIRED
############################################
MODEL_NAME="/model"
# Путь к директории базовой модели внутри контейнера.
# Compose монтирует туда RETRIEVER_DEPLOY_MODEL_PATH, обычно менять не нужно.

############################################
# LISTEN / AUTH
############################################
HOST="0.0.0.0"
# Адрес, на котором слушает API.
# 0.0.0.0 — слушать на всех интерфейсах контейнера.
# 127.0.0.1 — только локально внутри контейнера.

PORT="8000"
# Порт OpenAI-compatible API внутри контейнера.
# Наружу он пробрасывается через RETRIEVER_DEPLOY_PORT в .env, здесь менять не нужно.

API_KEY="${VLLM_API_KEY:-}"
# Bearer token, который сервер требует на /v1/*; пусто — без авторизации.
# Задаётся в .env (RETRIEVER_DEPLOY_API_KEY): compose передаёт его в контейнер как
# VLLM_API_KEY, им же ходит healthcheck. Здесь не меняй.
# Должен совпадать с ключом, с которым virag ходит в этот endpoint (RETRIEVER_API_KEY).

SERVED_MODEL_NAME="retriever-model"
# Имя модели, которое клиент (virag) указывает в поле "model".
# Должно совпадать с RETRIEVER_MODEL_NAME в настройках virag.
# Если смонтирован LoRA adapter, под этим именем отдаётся base+adapter,
# а сама база регистрируется под BASE_MODEL_NAME (см. LORA).

############################################
# RUNNER
############################################
RUNNER="pooling"
# Режим работы модели.
# pooling — выдача эмбеддингов через /v1/embeddings.
# Для embedding-моделей менять не нужно.

POOLER_CONFIG=""
# JSON для --pooler-config (тип pooling, normalize и т.п.).
# Пусто — использовать конфиг из самой модели.
# Пример: '{"pooling_type": "LAST", "normalize": true}'

############################################
# PARALLELISM
############################################
TENSOR_PARALLEL_SIZE="1"
# На сколько GPU распиливается одна модель по тензорному параллелизму.
# Для эмбеддинг-моделей (2B–8B) обычно хватает одной GPU.
# Увеличивай, только если модель не влезает в одну GPU.

PIPELINE_PARALLEL_SIZE="1"
# Пайплайновый параллелизм.
# Для обычных запусков оставляй 1.

############################################
# MODEL / MEMORY
############################################
MAX_MODEL_LEN="16384"
# Максимальная длина входа (в токенах), которую сервер резервирует и поддерживает.
# Для картиночных эмбеддингов учитывай, что одна страница = много image-токенов.
# Чем выше значение, тем больше расход памяти.
# Уменьшай, если сервер не стартует или слишком жрёт VRAM.
# Для 6 GB GPU: 4096.

GPU_MEMORY_UTILIZATION="0.9"
# Целевая доля VRAM, которую vLLM старается занять.
# Больше значение — больше места под активации и выше потенциальный throughput.
# Слишком высокое значение может привести к OOM или нестабильному старту.
# Если GPU делится с другими сервисами — снижай (например, 0.22).
# Для 6 GB GPU: 0.80.

MAX_NUM_SEQS="8"
# Максимальное число одновременно обрабатываемых запросов в scheduler.
# Больше — выше конкурентность, но выше пик памяти.
# Уменьшай, если упираешься в память. Для 6 GB GPU: 1.
# Меняй вместе с MAX_NUM_BATCHED_TOKENS, а не изолированно.

MAX_NUM_BATCHED_TOKENS="4096"
# Потолок суммарных токенов в одном scheduler batch.
# Больше — лучше утилизация GPU, но выше пики памяти.
# Меньше — стабильнее, но ниже throughput.
# Для pooling-моделей без chunked prefill не должен быть меньше MAX_MODEL_LEN.

SCHEDULING_POLICY="priority"
# Порядок обслуживания запросов в очереди scheduler'а.
# fcfs     — строго в порядке прихода.
# priority — по полю `priority` запроса (меньше — раньше), при равенстве по времени.
# virag шлёт priority только для фоновой загрузки (RETRIEVER_BULK_PRIORITY > 0),
# чтобы запросы из чата обгоняли очередь тайлов enroll/revision. Запросы без поля
# имеют priority 0, поэтому без настройки virag режим priority ведёт себя как fcfs.
# Ненулевой priority на сервере в режиме fcfs — ошибка (virag проверяет при старте).

############################################
# LORA
############################################
ENABLE_LORA="auto"
# auto — включить LoRA, если по пути ADAPTER_DIR лежит adapter_config.json.
# 1    — включить всегда (сервер упадёт, если адаптера нет).
# 0    — не добавлять LoRA flags, даже если адаптер смонтирован.
# vLLM не поддерживает DoRA — такой адаптер сначала нужно смёрджить в базу.

ADAPTER_DIR="/adapter"
# Путь к LoRA adapter внутри контейнера.
# Compose монтирует туда RETRIEVER_DEPLOY_ADAPTER_PATH. Если адаптер не нужен,
# укажи в .env путь к любой папке без adapter_config.json.

BASE_MODEL_NAME="base-model"
# Имя, под которым регистрируется база, пока LoRA включена.
# Клиентам оно не нужно: они используют SERVED_MODEL_NAME (base+adapter).

MAX_LORA_RANK="32"
# Максимальный rank LoRA.
# Должен покрывать rank подключаемого adapter'а.
# Чем выше rank, тем больше memory overhead.

############################################
# DTYPE / EXECUTION / QUANTIZATION
############################################
DTYPE="auto"
# Тип вычислений.
# auto — дать vLLM выбрать подходящий dtype.
# Можно задать явно: bfloat16 / float16, если нужно жёстко контролировать запуск.

QUANTIZATION=""
# Квантизация весов на лету.
# Пусто — полная точность.
# fp8 — вдвое меньше памяти под веса (нужна Ada/Hopper).
# Для 6 GB GPU с 2B моделью: fp8.

ENFORCE_EAGER="0"
# 1 — не захватывать CUDA graphs, экономит VRAM ценой скорости.
# 0 — обычный режим.
# Для 6 GB GPU: 1.

TRUST_REMOTE_CODE="0"
# Разрешение выполнять custom code из model repository.
# 0 — безопаснее, но некоторые модели не заведутся без этого.
# 1 — включай только если модель действительно этого требует и источник доверенный.

TOKENIZER_MODE="auto"
# Режим токенайзера.
# auto — fast tokenizer, если доступен.

LOAD_FORMAT="auto"
# Формат загрузки весов.
# auto — определить по содержимому директории модели.

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
# Resolve LoRA
############################################
LORA_ACTIVE="0"

if [[ "$ENABLE_LORA" == "1" ]]; then
  LORA_ACTIVE="1"
elif [[ "$ENABLE_LORA" == "auto" && -f "$ADAPTER_DIR/adapter_config.json" ]]; then
  LORA_ACTIVE="1"
fi

# Когда LoRA активна, база отдаётся под BASE_MODEL_NAME, а SERVED_MODEL_NAME
# достаётся adapter'у — так клиент прозрачно получает base+adapter.
if [[ "$LORA_ACTIVE" == "1" ]]; then
  BASE_SERVED_NAME="$BASE_MODEL_NAME"
else
  BASE_SERVED_NAME="$SERVED_MODEL_NAME"
fi

############################################
# Build argv
############################################
ARGS=()

# Core
ARGS+=("$MODEL_NAME")
ARGS+=("--host" "$HOST")
ARGS+=("--port" "$PORT")
ARGS+=("--served-model-name" "$BASE_SERVED_NAME")

# Auth
if [[ -n "$API_KEY" ]]; then
  ARGS+=("--api-key" "$API_KEY")
fi

# Runner
ARGS+=("--runner" "$RUNNER")

if [[ -n "$POOLER_CONFIG" ]]; then
  ARGS+=("--pooler-config" "$POOLER_CONFIG")
fi

# Parallelism
ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
ARGS+=("--pipeline-parallel-size" "$PIPELINE_PARALLEL_SIZE")

# Model / memory
ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
ARGS+=("--scheduling-policy" "$SCHEDULING_POLICY")

# LoRA
if [[ "$LORA_ACTIVE" == "1" ]]; then
  ARGS+=("--enable-lora")
  ARGS+=("--max-lora-rank" "$MAX_LORA_RANK")
  ARGS+=("--lora-modules" "${SERVED_MODEL_NAME}=${ADAPTER_DIR}")
fi

# Dtype / execution
ARGS+=("--dtype" "$DTYPE")
ARGS+=("--tokenizer-mode" "$TOKENIZER_MODE")
ARGS+=("--load-format" "$LOAD_FORMAT")

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
echo "pooler config:           $([[ -n "$POOLER_CONFIG" ]] && echo "$POOLER_CONFIG" || echo model default)"
echo
echo "tensor parallel:         $TENSOR_PARALLEL_SIZE"
echo "pipeline parallel:       $PIPELINE_PARALLEL_SIZE"
echo
echo "max model len:           $MAX_MODEL_LEN"
echo "gpu mem utilization:     $GPU_MEMORY_UTILIZATION"
echo "max num seqs:            $MAX_NUM_SEQS"
echo "max batched tokens:      $MAX_NUM_BATCHED_TOKENS"
echo "scheduling policy:       $SCHEDULING_POLICY"
echo
echo "lora:                    $([[ "$LORA_ACTIVE" == "1" ]] && echo enabled || echo disabled)"
if [[ "$LORA_ACTIVE" == "1" ]]; then
  echo "lora adapter:            $ADAPTER_DIR (max rank $MAX_LORA_RANK)"
  echo "base served name:        $BASE_SERVED_NAME"
  echo "client model name:       $SERVED_MODEL_NAME (base+adapter)"
fi
echo
echo "dtype:                   $DTYPE"
echo "quantization:            $([[ -n "$QUANTIZATION" ]] && echo "$QUANTIZATION" || echo disabled)"
echo "enforce eager:           $([[ "$ENFORCE_EAGER" == "1" ]] && echo enabled || echo disabled)"
echo "trust remote code:       $([[ "$TRUST_REMOTE_CODE" == "1" ]] && echo enabled || echo disabled)"
echo
echo "uvicorn log level:       $UVICORN_LOG_LEVEL"
echo "OpenAI endpoint:         /v1/embeddings"
echo "metrics:                 /metrics"
echo
# The key never reaches the log: its value is masked in the printed command.
SHOWN=("${ARGS[@]}")
for i in "${!SHOWN[@]}"; do
  [[ "${SHOWN[$i]}" == "--api-key" ]] && SHOWN[$((i + 1))]="***"
done

echo "full command:"
printf ' %q' vllm serve "${SHOWN[@]}"
echo
echo

exec vllm serve "${ARGS[@]}"
