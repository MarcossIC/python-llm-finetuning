import argparse
import os
import logging
import random
import gc
import psutil
import numpy as np
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import evaluate
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM with QLoRA and optimizations"
    )
    parser.add_argument("--model_name", type=str,
                        default="./Llama-3.2-1B/Llama-3.2-1B/",
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
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
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
    return parser.parse_args()



def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_system_info(logger):
    """Log system information"""
    if torch.cuda.is_available():
        logger.info("📊 System Information:")
        logger.info(f"  - CUDA Version: {torch.version.cuda}")
        logger.info(f"  - GPU: {torch.cuda.get_device_name(0)}")
        memory_info = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  - GPU Memory: {memory_info:.2f} GB")
        logger.info(f"  - RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        logger.info(f"  - PyTorch Version: {torch.__version__}")


def log_gpu_memory(logger):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"🧠 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def free_memory():
    """Free unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_and_format(path, logger):
    logger.info(f"📥 Loading dataset from {path}...")
    raw = load_dataset("json", data_files=path, split="train")
    
    logger.info(f"📊 Dataset info: {len(raw)} samples")
    
    # Display sample from dataset
    if len(raw) > 0:
        logger.info("📝 Sample from raw dataset:")
        logger.info(str(raw[0])[:500] + "..." if len(str(raw[0])) > 500 else str(raw[0]))

    def format_example(ex):
        messages = ex.get("messages", []) or []
        prompt = ""
        for m in messages:
            tag = "<|user|>" if m.get("role") == "user" else "<|assistant|>"
            prompt += f"{tag}\n{m.get('content', '')}\n"
        return {"text": prompt.strip()}

    formatted = raw.map(format_example, remove_columns=raw.column_names)
    
    # Display formatted sample
    if len(formatted) > 0:
        logger.info("📝 Sample after formatting:")
        logger.info(str(formatted[0])[:500] + "..." if len(str(formatted[0])) > 500 else str(formatted[0]))
    
    return formatted

# MemoryLoggingCallback
class MemoryLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger = logging.getLogger(__name__)
            logger.info(f"Step {state.global_step}: 🧠 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def main():
    # 1. Parseas argumentos
    args = parse_args()
    # 2. Siembras todas las fuentes de aleatoriedad
    set_seed(args.seed)
    # 3. Ahora configuras logging, cargas datos, modelos, etc.
    logger = setup_logging()

    # Determine device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    logger.info(f"🖥️  Using device: {device_str}")
    device = torch.device(device_str)

    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        logger.error(f"❌ Dataset not found at {args.dataset_path}")
        return

    # Creating output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        logger.info(f"📁 Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    # Load and format dataset
    dataset = load_and_format(args.dataset_path, logger)

    logger.info("🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning("⚠️ Tokenizador no tenía pad_token, se estableció a eos_token.")
            else:
                 # Añade un pad token si ni siquiera tiene eos (raro pero posible)
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 logger.warning("⚠️ Tokenizador no tenía pad_token ni eos_token. Se añadió [PAD]")
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {e}")
        return

    # Load model with quantization
    logger.info(f"⚙️  Loading model with {args.bit_precision}-bit quantization...")
    
   # Load model on CPU without BitsAndBytes quantization
    try:
        torch_dtype = torch.float32  # Use float32 for CPU
    
    # Set device_map to CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="cpu",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Help with memory usage on CPU
        )
        
        use_grad_ckpt = False # O False si decides desactivarlo en TrainingArguments

        # Asegúrate de que la configuración de caché del modelo sea compatible
        # con gradient checkpointing si está activado.
        # prepare_model_for_kbit_training lo hace, pero puedes ser explícito:
        if use_grad_ckpt:
            base_model.config.use_cache = False # Necesario para gradient checkpointing

        # Llama a la función de preparación
        model_prepared = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=use_grad_ckpt
        )
        
        
        logger.info("✅ Base model loaded successfully on CPU")
    
    # Log memory usage after model load
        # log_gpu_memory(logger)


        target_modules = None
    
    # Try to detect model architecture and set appropriate target modules
        if hasattr(model_prepared, "config"):
            model_type = getattr(base_model.config, "model_type", "")
            if "llama" in model_type.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "mistral" in model_type.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "gpt-neox" in model_type.lower():
                target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            elif "gpt2" in model_type.lower():
                target_modules = ["c_attn", "c_proj", "c_fc"]
            else:
                # Default fallback
                target_modules = ["query", "key", "value", "dense"]

        if target_modules is None:
            logger.warning("⚠️ Couldn't determine target modules, using default set")
            target_modules = ["query", "key", "value", "dense"]

        logger.info(f"🎯 Using target modules: {target_modules}")

    # Remove GPU-specific optimizations
        # base_model.config.use_cache = False
    
    # Configure LoRA without prepare_model_for_kbit_training
        logger.info(f"⚙️ Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        # lora_cfg = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_alpha,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        #     lora_dropout=args.lora_dropout,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # for param in base_model.parameters():
        #     param.requires_grad = False
    
        model = get_peft_model(model_prepared, lora_cfg)
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise ValueError("No trainable parameters found! Training can't proceed.")
        logger.info(f"✅ PEFT model created - Trainable parameters: {model.print_trainable_parameters()}")
    
    # Log memory after PEFT setup
        log_gpu_memory(logger)
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return

    logger.info("🔍 Tokenizing dataset (multiprocessing)...")
    def tokenize_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=min(os.cpu_count() or 1, 4),
        remove_columns=["text"]
    )
        
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    logger.info(f"✅ Dataset tokenized - {len(tokenized)} samples")

    # Use a data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Prepare evaluation split
    splits = tokenized.train_test_split(test_size=0.1)
    train_ds, eval_ds = splits["train"], splits["test"]

    # Free memory
    free_memory()
    log_gpu_memory(logger)

    # PyTorch 2.0 compile (if available)
    #if hasattr(torch, 'compile') and torch.cuda.is_available():
    #    logger.info("⚡️ Compiling model with torch.compile...")
    #    try:
    #        model = torch.compile(model)
    #        logger.info("✅ Model compiled successfully")
    #    except Exception as e:
    #        logger.warning(f"⚠️ Model compilation failed, continuing without compilation: {e}")

    logger.info("⚙️ Configuring training parameters...")
    # Configure training arguments dynamically
    common_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
        gradient_checkpointing=use_grad_ckpt,
        report_to=[],
        logging_dir=os.path.join(args.output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
    )
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MemoryLoggingCallback(), EarlyStoppingCallback(early_stopping_patience=2)]
    )

    logger.info("🚀 Starting training...")
    try:
        free_memory()
        log_gpu_memory(logger)
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