import argparse
import os
from os.path import exists, join, isdir
import logging
import random
import gc
import re
import psutil
import numpy as np
import importlib.metadata
import platform
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)

from peft.tuners.lora import LoraLayer
from rich.logging import RichHandler
from rich.console import Console
from packaging import version


console = Console()
cpu_brand = platform.processor()

# Paresea y recupera los argumentos del usuario
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM with QLoRA and optimizations"
    )
    parser.add_argument("--model_name", type=str,
                        default="./deepseek-math-7b-instruct",
                        help="Path or name of the pretrained model")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="JSON file with training data")
    parser.add_argument("--output_dir", type=str, default="finetuned-math-model",
                        help="Where to save the finetuned model")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max token length for inputs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Steps between checkpoint saves")
    parser.add_argument("--bit_precision", type=int, default=4, choices=[4, 8],
                        help="Quantization precision (4 or 8 bits)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether to push to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hub model ID (user/model)")
    
    parser.add_argument("--full_finetune", action="store_true",
                        help="If set, do full fine-tuning instead of LoRA adapters.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision if available.")
    return parser.parse_args()

# Optimiza hiperparámetros según el dispositivo y memoria disponible
def optimize_hyperparams(args, logger):
    """
    Ajusta automáticamente hiperparámetros según el dispositivo y memoria disponible.
    """
    using_gpu = torch.cuda.is_available() and args.device != "cpu"
    args.using_gpu = using_gpu
    optimized = args.__dict__.copy()
    
    # Ajuste basado en si usamos GPU o CPU
    if using_gpu:
        # En GPU, podemos ser más agresivos con batch size y lr
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_mem = max(1, vram_total - 1)  # Dejar 1 GB libre
        avail_cpu = psutil.virtual_memory().available / (1024 ** 3)
        cpu_mem = max(1, avail_cpu - 2)  # Dejar 2 GB libre

        logger.info(f"🖥️ GPU detected, memory budgets → GPU: {gpu_mem:.1f} GiB, CPU: {cpu_mem:.1f} GiB")

        args.gpu_mem = gpu_mem
        args.cpu_mem = cpu_mem
        args.gradient_checkpointing = True
        
        # Ajustes basados en VRAM disponible
        if vram_total >= 24:  # GPU de alta gama (24+ GB)
            logger.info("🚀 High-end GPU detected, optimizing for performance")
            optimized['batch_size'] = max(args.batch_size, 4)
            optimized['grad_accum_steps'] = min(args.grad_accum_steps, 4)
        elif vram_total >= 10:  # GPU de gama media (10-24 GB)
            logger.info("🚀 Mid-range GPU detected, balancing performance and memory")
            optimized['batch_size'] = max(args.batch_size, 2)
            optimized['grad_accum_steps'] = min(args.grad_accum_steps, 8)
        else:  # GPU de gama baja (<10 GB)
            logger.info("🔧 Limited GPU memory detected, optimizing for stability")
            optimized['batch_size'] = min(args.batch_size, 1)
            optimized['grad_accum_steps'] = max(args.grad_accum_steps, 16)
            optimized['max_length'] = min(args.max_length, 512)  # Reducir longitud de secuencia
            
        # Common GPU optimizations
        optimized['bit_precision'] = args.bit_precision  # Mantener la cuantización solicitada
    else:
        # En CPU priorizar estabilidad sobre velocidad
        logger.info("🔧 CPU detected, optimizing for stability over speed")
        optimized['batch_size'] = 1
        optimized['grad_accum_steps'] = max(args.grad_accum_steps, 16)
        optimized['max_length'] = min(args.max_length, 256)  # Secuencias más cortas
        optimized['lr'] = min(args.lr, 1e-4)  # LR más conservador
        optimized['lora_r'] = min(args.lora_r, 4)  # Reducir rank para menos parámetros
        optimized['bf16'] = False
        args.gradient_checkpointing = False

    # Calcular procesos de carga de datos
    args.num_proc = 1 if not using_gpu else min(os.cpu_count() or 1, 4)
    logger.info(f"🔢 Number of dataloader processes: {args.num_proc}")

    # Asegurar campos básicos
    if not hasattr(args, 'full_finetune'):
        optimized['full_finetune'] = False

    if not hasattr(args, 'bf16'):
        optimized['bf16'] = using_gpu  # Si tenemos GPU, activamos bf16 si posible

    # Log de los cambios realizados
    changes = {k: (getattr(args, k), v) for k, v in optimized.items() 
              if hasattr(args, k) and getattr(args, k) != v}
    
    if changes:
        logger.info("📊 Hyperparameter optimizations:")
        for param, (old, new) in changes.items():
            logger.info(f"  • {param}: {old} → {new}")
    
    # Actualizar args con valores optimizados
    for k, v in optimized.items():
        setattr(args, k, v)
        
    return args

# Configura el logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(__name__)

# Siembra la semilla para reproducibilidad
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_ipex_available(logger):
    """
    Verifica si Intel Extension for PyTorch (IPEX) está instalada y si su
    versión mayor/menor coincide con la de PyTorch.
    """
    try:
        # Intenta obtener ambas versiones
        torch_version_str = importlib.metadata.version("torch")
        ipex_version_str = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        # Si torch o ipex no están, IPEX no está "disponible"
        return False
    try:
        # Parsear las versiones
        torch_ver = version.parse(torch_version_str)
        ipex_ver = version.parse(ipex_version_str)
        # Comparar directamente los componentes mayor y menor
        if (torch_ver.major, torch_ver.minor) != (ipex_ver.major, ipex_ver.minor):
            logger.warning(
                f"Intel Extension for PyTorch {ipex_ver.major}.{ipex_ver.minor} necesita PyTorch {ipex_ver.major}.{ipex_ver.minor}.*, "
                f"pero se encontró PyTorch {torch_version_str}. Cambia a la versión correspondiente y ejecuta de nuevo."
            )
            return False # Versiones no compatibles

        # Si llegamos aquí, está instalado y las versiones base coinciden
        return True

    except version.InvalidVersion:
        # En el caso improbable de que una versión no se pueda parsear
        logger.warning(
            f"No se pudieron parsear las cadenas de versión: Torch='{torch_version_str}', IPEX='{ipex_version_str}'. "
            "No se puede determinar la compatibilidad."
            )
        return False

# MemoryLoggingCallback
class MemoryLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger = logging.getLogger(__name__)
            logger.info(f"Step {state.global_step}: 🧠 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

#log_system_info
def log_gpu_memory(logger):
    """Log GPU memory usage for all available GPUs"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"GPU {i}: Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB")
            except Exception as e:
                logger.warning(f"Could not log memory for GPU {i}: {str(e)}")
    else:
        logger.info("No GPU available")

# Obtiene los módulos de modelo según la arquitectura
def get_model_targets(model_config, logger):
    # Elegir módulos según arquitectura
    arch = getattr(model_config, "model_type", "").lower()
    if 'llama' in arch or 'mistral' in arch:
        targets = ['q_proj','k_proj','v_proj','o_proj']
    elif 'neox' in arch:
        targets = ['query_key_value','dense','dense_h_to_4h','dense_4h_to_h']
    elif 'gpt2' in arch:
        targets = ['c_attn','c_proj','c_fc']
    else:
        targets = ['query','key','value','dense']

    if targets is None:
        logger.warning("⚠️ Couldn't determine target modules, using default set")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Common default
    else:
        logger.info(f"🎯 Using LoRA target_modules={targets}")
    return targets

# Libera memoria no utilizada
def free_memory():
    """Free unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Carga y prepara un modelo para CPU o GPU con manejo robusto de errores
def load_model(args, logger):
    """
    Carga y prepara un modelo para CPU o GPU con manejo robusto de errores.
    """
    logger.info(f"⚙️ Loading model with {getattr(args, 'bit_precision', 'full')} precision...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning("⚠️ pad_token set to eos_token.")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning("⚠️ Added [PAD] token.")
        logger.info("✅ Tokenizer ready")
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {str(e)}")
        raise

    try:        
        if args.using_gpu:            
            # 4 niveles de degradación para GPU
            strategies = [
                # Estrategia 1: GPU completa con cuantización, sin offload
                lambda: _load_gpu_model(args, logger, offload=False, fp32_cpu_offload=False),
                # Estrategia 2: GPU + CPU offload con soporte fp32 específico
                lambda: _load_gpu_model(args, logger, offload=True, fp32_cpu_offload=True),
                # Estrategia 3: GPU + CPU con offload pesado y cuantización máxima
                lambda: _load_gpu_model(args, logger, offload=True, fp32_cpu_offload=True, 
                                       force_max_quantization=True),
                # Estrategia 4: CPU-fallback como último recurso
                lambda: _load_cpu_model(args, logger)
            ]
            last_error = None
            for i, strategy in enumerate(strategies):
                try:
                    logger.info(f"🔄 Trying model loading strategy {i+1}/{len(strategies)}...")
                    model = strategy()
                    logger.info(f"✅ Successfully loaded model with strategy {i+1}")
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"⚠️ Strategy {i+1} failed: {str(e)}")
                    free_memory()  # Liberar memoria antes de intentar siguiente estrategia
                    
                    # Si llegamos a la última estrategia y también falla, propagar el error
                    if i == len(strategies) - 1:
                        logger.error(f"❌ All loading strategies failed. Last error: {str(last_error)}")
                        raise last_error
        else:
            # CPU-only: full precision
            model = _load_cpu_model(args, logger)

        # 3) Preparar para k-bit training
        if not getattr(args, 'full_finetune', False):
            if getattr(args, 'bit_precision', None) in (4,8):
                model = prepare_model_for_kbit_training(model)

        # 4) Configurar LoRA o cargar adapters desde checkpoint
        if not getattr(args, 'full_finetune', False):
            if getattr(args, 'checkpoint_dir', None):
                logger.info("🔄 Loading adapters from checkpoint...")
                model = PeftModel.from_pretrained(
                    model, 
                    args.checkpoint_dir, 
                    is_trainable=True
                )
                logger.info("✅ Adapters loaded")
            else:
                logger.info(f"⚙️ Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
                targets = get_model_targets(model.config, logger)

                lora_cfg = LoraConfig(
                    r = args.lora_r,
                    lora_alpha = args.lora_alpha,
                    target_modules = targets,
                    lora_dropout = args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_cfg)
                logger.info("✅ LoRA applied")

            # (Opcional, pero recomendado) Ajustar módulos de precisión
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if getattr(args, 'bf16', False):
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if getattr(args, 'bf16', False) and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())

            logger.info(f"📊 Params: Trainable: {trainable} || Total: {total} || Use: ({100*trainable/total:.2f}%)")

            if trainable == 0:
                logger.warning("⚠️ No trainable parameters found! Check LoRA configuration.")
        else:
            logger.info("🔄 No LoRA applied")

        log_gpu_memory(logger)
        # Redimensionar embeddings si se añadieron tokens
        if len(tokenizer) != model.config.vocab_size:
            logger.info(f"🔄 Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

def _load_gpu_model(args, logger, offload=False, fp32_cpu_offload=False, force_max_quantization=False):
    """
    Helper function para cargar modelo en GPU con diferentes opciones.
    """
    # Configurar cuantización
    bit_precision = 4 if force_max_quantization else getattr(args, 'bit_precision', 16)
    quant_cfg = None
    
    compute_dtype = torch.float16 if args.bit_precision in (4,8) else torch.float32
    # Debería considerar args.bf16 también
    if compute_dtype == torch.float16 and args.bit_precision == 4:
        if args.bf16 and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        elif torch.cuda.is_bf16_supported():
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            
    if compute_dtype == torch.float16 and (is_ipex_available(logger) and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        logger.info('Intel XPU does not support float16 yet, so switching to bfloat16')

    if bit_precision in (4, 8):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=(bit_precision == 4),
            load_in_8bit=(bit_precision == 8),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=(not force_max_quantization),
            bnb_4bit_quant_type = 'nf4',
            llm_int8_enable_fp32_cpu_offload=fp32_cpu_offload  # Clave para fixear el error
        )
    
    # Detect device and distribution
    device_map = 'auto' if offload else 'cuda:0'

    # Calcular presupuestos de memoria
    opts = dict(
        quantization_config = quant_cfg,
        device_map = device_map,
        max_memory = ({0: f"{args.gpu_mem}GiB", 'cpu': f"{args.cpu_mem}GiB"} if offload else {0: f"{args.gpu_mem}GiB"}),
        torch_dtype = compute_dtype,
        trust_remote_code = True,
    )
    
    # Configurar memory map según offload
    if offload:
        opts['offload_folder'] = 'offload'
        opts['offload_state_dict'] = True
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **opts)

    if quant_cfg:
        model = prepare_model_for_kbit_training(model)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    return model

def _load_cpu_model(args, logger):
    """
    Helper function para cargar modelo en CPU.
    """
    logger.info("🌐 Loading model in CPU-only mode with float32 precision...")
    logger.warning("⚠️ CPU training will be slow. Consider using a GPU if available.")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={'': 'cpu'},
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info("✅ Model loaded on CPU (float32)")
    model = prepare_model_for_kbit_training(model)
    
    return model

def get_last_checkpoint(checkpoint_dir, logger, is_full_finetune=False):
    # Verificar si el directorio existe
    if not isdir(checkpoint_dir):
        logger.debug(f"Directorio de checkpoint no encontrado: {checkpoint_dir}")
        return None, False
    
    # Verificar si el entrenamiento ya fue marcado como completado
    completed_flag = join(checkpoint_dir, "COMPLETED")
    if exists(completed_flag):
        logger.info("Entrenamiento previamente completado")
        return None, True
    
    required_files = {
        "common": ["training_args.bin"],
        "peft": ["adapter_config.json", "adapter_model.safetensors"],
        "full": ["pytorch_model.bin", "config.json", "model.safetensors"]
    }
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")
    valid_checkpoints  = []

    try:
        # Buscar todos los directorios de checkpoint válidos
        for entry in os.listdir(checkpoint_dir):
            entry_path = join(checkpoint_dir, entry)
            
            if os.path.isdir(entry_path) and checkpoint_pattern.match(entry):
                existing_files = set(os.listdir(entry_path))
                
                # Verificar archivos según tipo de modelo
                if not is_full_finetune:
                    required = required_files["common"] + required_files["peft"]
                else:
                    required = required_files["common"] + required_files["full"]
                
                # Aceptar diferentes formatos de pesos
                weight_files = {"pytorch_model.bin", "model.safetensors", "adapter_model.safetensors"}
                has_weights = len(weight_files & existing_files) > 0
                
                if has_weights and all(f in existing_files for f in required):
                    try:
                        step = int(checkpoint_pattern.match(entry).group(1))
                        valid_checkpoints.append((step, entry_path))
                    except ValueError:
                        continue
        
        # Obtener el último checkpoint
        if valid_checkpoints:
            valid_checkpoints.sort(key=lambda x: x[0])
            last_checkpoint = valid_checkpoints[-1][1]
            logger.info(f"Found valid checkpoint: {last_checkpoint}")
            return last_checkpoint, False

        logger.debug("No valid checkpoints found")
        return None, False
    except Exception as e:
        logger.error(f"Error buscando checkpoints: {str(e)}")
        return None, False  

def process_dataset(args, tokenizer, logger):
    """
    Procesa dataset con manejo optimizado de memoria y rendimiento.
    """
    logger.info(f"📥 Loading dataset from {args.dataset_path}...")
    
    try:
        # Cargar dataset crudo
        raw = load_dataset("json", data_files=args.dataset_path, split="train")
        logger.info(f"📊  Dataset loaded: {len(raw)} samples")
        
        # Configurar chat template si es necesario
        if tokenizer.chat_template is None:
            logger.warning("⚠️ No chat template found, setting default...")
            # Template simple que concatena mensajes
            tokenizer.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"
        else:
            logger.info("✅ Using model's built-in chat template")

        # Mostrar ejemplo de muestra
        if len(raw) > 0:
            logger.info("📝  Sample from raw dataset:")
            sample = raw[0].get("messages", [])
            for m in sample:
                role = m.get("role", "").upper()
                content = m.get("content", "")
                logger.info(f"  • {role}: {content}")
        
        # Verificar estructura del dataset
        required_keys = ["messages"]
        sample = raw[0] if len(raw) > 0 else {}
        missing_keys = [k for k in required_keys if k not in sample]
        
        if missing_keys:
            logger.warning(f"⚠️  Dataset missing required keys: {missing_keys}")
            logger.warning("⚠️  Expected format: {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}")
            raise ValueError(f"Dataset missing required keys: {missing_keys}")
        
        # Formatear ejemplos según el formato de mensajes
        def format_example(ex):
            messages = ex.get("messages", []) or []
            # Validar estructura de mensajes
            if not messages:
                logger.warning("⚠️  Sample with empty messages found")
                return {"text": ""}
                
            prompt = ""
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                
                if not role or not content:
                    continue
                    
                tag = "<|user|>" if role == "user" else "<|assistant|>"
                prompt += f"{tag}\n{content}\n"
            
            return {"text": prompt.strip()}
        
        # Usar diferentes estrategias según el dispositivo
        if args.using_gpu and len(raw) > 10000:
            # procesar en batches para evitar OOM
            logger.info("🔄  Large dataset detected, processing in batches...")
            formatted = raw.map(
                format_example, 
                batched=True,
                batch_size=1000,
                remove_columns=raw.column_names,
                num_proc=min(os.cpu_count() or 1, 4),
                desc="Formatting dataset"
            )
        else:
            formatted = raw.map(
                format_example,
                remove_columns=raw.column_names,
                num_proc=1 if not args.using_gpu else min(os.cpu_count() or 1, 4),
                desc="Formatting dataset"
            )
        
        # Verificar ejemplos vacíos
        empty_samples = sum(1 for ex in formatted if not ex["text"])
        if empty_samples > 0:
            logger.warning(f"⚠️  Found {empty_samples} empty samples after formatting")
            # Filtrar ejemplos vacíos
            formatted = formatted.filter(lambda ex: bool(ex["text"]), desc="Filtering empty samples")
            logger.info(f"✅  Kept {len(formatted)} valid samples after filtering")
        
        # Mostrar ejemplo formateado
        if len(formatted) > 0:
            logger.info("📝  Sample after formatting:")
            truncated = str(formatted[0])
            logger.info(truncated[:500] + "..." if len(truncated) > 500 else truncated)
        
        # Tokenizar el dataset
        logger.info(f"🔤  Tokenizing dataset (max_length={args.max_length})...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_attention_mask=True,
                return_tensors=None
            )
        
        # Tokenizar con progress bar
        tokenized = formatted.map(
            tokenize_function,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        # Convertir a formato PyTorch
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # Verificación de calidad
        if len(tokenized) == 0:
            raise ValueError("Dataset vacío después del procesamiento. Verifica el formato de entrada.")
            
        logger.info(f"✅  Procesamiento completo: {len(tokenized)} muestras válidas")
        
        # Verificar distribución de longitudes
        lengths = [len(x) for x in tokenized["input_ids"]]
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        logger.info(f"📊  Estadísticas de longitud: promedio={avg_len:.1f}, máximo={max_len}")
        
        # Advertir si hay muchas secuencias truncadas
        if max_len >= args.max_length:
            truncated_pct = sum(1 for l in lengths if l >= args.max_length) / len(lengths) * 100
            logger.warning(f"⚠️ {truncated_pct:.1f}% de secuencias fueron truncadas al máximo ({args.max_length})")
        
        # Split de train/eval
        splits = tokenized.train_test_split(test_size=min(0.1, 1000/len(tokenized)))
        train_ds, eval_ds = splits["train"], splits["test"]
        
        logger.info(f"✅  Dataset preparado: {len(train_ds)} muestras de entrenamiento, {len(eval_ds)} de evaluación")
        
        return train_ds, eval_ds
        
    except Exception as e:
        logger.error(f"❌  Error procesando dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    # 1. Parseas argumentos
    args = parse_args()
    # 2. Siembras todas las fuentes de aleatoriedad
    set_seed(args.seed)
    # 3. Ahora configuracion y carga de datos, modelos, etc.
    logger = setup_logging()
    # 5 Optimizar hiperparámetros
    args = optimize_hyperparams(args, logger)

    if "Intel" in cpu_brand:
        logger.info("🏎️  Detected Intel CPU, can use IPEX if installed.")
    elif "AMD" in cpu_brand:
        logger.info("🚀  Detected AMD CPU, using native optimizations (no IPEX available).")
    else:
        logger.info("🤔  Unknown CPU, using default PyTorch settings.")

    if args.using_gpu:
        logger.info("✅  Enabling GPU optimizations for allow_tf32")
        torch.backends.cuda.matmul.allow_tf32 = True

    # Pre-checks
    if not os.path.exists(args.dataset_path):
        logger.error(f"❌ Dataset not found at {args.dataset_path}")
        return
    os.makedirs(args.output_dir, exist_ok=True)
        
    # Load model with quantization
    logger.info(f"🔤  Loading model and tokenizer Start...")
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir, logger, args.full_finetune)
    if completed_training:
        logger.info('🔤  Detected that training was already completed!')

    args.checkpoint_dir = checkpoint_dir
    args.completed_training = completed_training
    model, tokenizer = load_model(args, logger)
    if not args.using_gpu and is_ipex_available(logger):
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model)
        logger.info("✅  Intel Extension for PyTorch enabled (CPU optimizations)")
    else:
        logger.info("❌  No Intel Extension for PyTorch available")

    if args.using_gpu:
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # Determine device
    device = torch.device("cuda" if args.using_gpu else "cpu")
    logger.info(f"🖥️  Using device: {device}")

    # Procesa el dataset con el tokenizador
    train_ds, eval_ds = process_dataset(args, tokenizer, logger)

    # Free memory
    free_memory(); log_gpu_memory(logger)
    num_proc = 1 if not args.using_gpu else min(os.cpu_count() or 1, 4)
    logger.info("⚙️ Configuring training parameters...")
    # Configure training arguments dynamically
    common_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        use_cpu=not args.using_gpu
    )
    if args.using_gpu:
        common_args.update(fp16=True, optim='adamw_torch',
                           gradient_checkpointing=True,
                           dataloader_num_workers=num_proc)
    else:
        common_args.update(fp16=False, optim='adamw_torch',
                           gradient_checkpointing=False,
                           dataloader_num_workers=0)
    if args.push_to_hub:
        common_args.update(
            push_to_hub=True,
            hub_model_id=args.hub_model_id,
            hub_strategy="every_save",
        )

    training_args = TrainingArguments(**common_args)

    # Metrics
    perplexity_metric = evaluate.load("perplexity")
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = labels[..., 1:].reshape(-1)
        return {"perplexity": perplexity_metric.compute(predictions=shift_logits, references=shift_labels)["perplexity"]}

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MemoryLoggingCallback(), EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("🚀 Starting training...")
    try:
        free_memory(); log_gpu_memory(logger)
        trainer.train()
        trainer.save_model()
        if args.push_to_hub:
            trainer.push_to_hub()
        logger.info(f"✅ Training complete! Model saved to {args.output_dir}")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        # Save model in case of error
        logger.info("🔄 Attempting to save model before exit...")
        try:
            model.save_pretrained(os.path.join(args.output_dir, "checkpoint-error"))
            logger.info("✅ Emergency save completed")
        except Exception as save_error:
            logger.error(f"❌ Could not perform emergency save: {save_error}")


if __name__ == "__main__":
    main()