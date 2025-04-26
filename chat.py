import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # BitsAndBytesConfig eliminado
from peft import PeftModel
# import evaluate # Eliminado si no se usa
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import warnings
import psutil # Para info de RAM

# Ignorar advertencias específicas si son muy ruidosas (opcional)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Passing `pad_token_id` is deprecated.*")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat interactivo con modelo base + adaptadores LoRA en CPU"
    )
    parser.add_argument("--base_model", type=str,
                        default="./Llama-3.2-1B/Llama-3.2-1B/",
                        help="Ruta o nombre del modelo base")
    parser.add_argument("--adapter_dir", type=str,
                        default="finetuned-math-model",
                        help="Directorio con los adaptadores LoRA (.save_pretrained)")
    # --bit_precision ya no se usa para BitsAndBytes, pero se mantiene por si se implementa otra cuantización CPU
    # parser.add_argument("--bit_precision", type=int, choices=[4, 8], default=0,
    #                     help="Precisión de cuantización (0 para desactivar en CPU)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"], # Forzar CPU
                        help="Dispositivo (solo 'cpu' soportado en este script)")
    parser.add_argument("--max_new_tokens", type=int, default=256, # Reducido por defecto para CPU
                        help="Máximo de tokens a generar por respuesta")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperatura de muestreo")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--max_history", type=int, default=3, # Reducido por defecto para CPU
                        help="Número máximo de turnos de conversación a mantener")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)] # show_path=False para logs más limpios
    )
    return logging.getLogger(__name__)

def log_system_memory(logger):
    """Log system RAM usage"""
    try:
        ram_info = psutil.virtual_memory()
        used_gb = ram_info.used / (1024**3)
        total_gb = ram_info.total / (1024**3)
        logger.info(f"🧠 System RAM: {used_gb:.2f}GB used / {total_gb:.2f}GB total ({ram_info.percent}%)")
    except Exception as e:
        logger.warning(f"No se pudo obtener el uso de memoria RAM: {e}")

def main():
    args = parse_args()
    logger = setup_logging()

    # --- Forzar dispositivo a CPU ---
    effective_device = torch.device("cpu")
    logger.info(f"🖥️  Ejecutando exclusivamente en CPU. La inferencia puede ser lenta.")
    log_system_memory(logger)

    # --- Cuantización Desactivada para CPU ---
    quant_cfg = None # BitsAndBytes no se usa en CPU
    logger.info("🚫 Cuantización BitsAndBytes desactivada (optimización de GPU).")

    # --- Carga del Tokenizador ---
    try:
        logger.info(f"🔄 Cargando tokenizador desde: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning("⚠️ Tokenizador no tenía pad_token, se estableció a eos_token.")
            else:
                new_pad_token = '[PAD]'
                tokenizer.add_special_tokens({'pad_token': new_pad_token})
                logger.warning(f"⚠️ Tokenizador no tenía pad_token ni eos_token. Se añadió '{new_pad_token}'.")
                # Si se añadió un token nuevo, puede ser necesario redimensionar embeddings
                # Esto se haría antes de cargar el modelo si el modelo no lo maneja
                # base_model_config.vocab_size = len(tokenizer) # Ejemplo

        tokenizer.padding_side = "left"
        logger.info("✅ Tokenizador cargado.")

    except Exception as e:
        logger.error(f"❌ Error cargando el tokenizador: {e}", exc_info=True)
        return

    # --- Carga del Modelo Base en CPU ---
    try:
        logger.info(f"🔄 Cargando modelo base en CPU desde: {args.base_model} (puede tardar y usar RAM)...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=None, # Sin cuantización BnB
            device_map="cpu",        # Forzar carga en CPU
            torch_dtype=torch.float32, # Usar float32 para CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True # Intentar usar menos RAM durante la carga
        )
        logger.info(f"✅ Modelo base cargado en {base.device}.")
        log_system_memory(logger)

    except Exception as e:
        logger.error(f"❌ Error cargando el modelo base: {e}", exc_info=True)
        return

    # --- Carga de Adaptadores LoRA en CPU ---
    try:
        logger.info(f"🔄 Aplicando adaptadores LoRA desde: {args.adapter_dir}")
        # PeftModel hereda el device_map del modelo base
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        model.eval() # ¡Importante! Poner en modo evaluación
        logger.info(f"✅ Modelo PEFT listo en {model.device}.")
        log_system_memory(logger)
    except Exception as e:
        logger.error(f"❌ Error cargando los adaptadores LoRA: {e}", exc_info=True)
        return


    # --- Ciclo de Chat Interactivo ---
    logger.info(Panel(f"[bold green]Chat interactivo listo![/bold green]\nModelo: {args.base_model} + {args.adapter_dir}\nParámetros: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, max_new={args.max_new_tokens}, history={args.max_history}\nEscribe 'exit' o 'quit' para salir.", title="Configuración", border_style="blue"))

    history = [] # Lista de diccionarios: {'role': 'user'/'assistant', 'content': '...'}

    try:
        while True:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
            if user_input.strip().lower() in ('exit', 'quit'):
                console.print("[bold yellow]Saliendo...[/bold yellow]")
                break

            history.append({"role": "user", "content": user_input})

            # --- Construcción del Prompt ---
            prompt_applied = False
            full_prompt_string = ""
            try:
                # Intentar usar plantilla de chat (preferido)
                full_prompt_string = tokenizer.apply_chat_template(
                    history,
                    tokenize=False,
                    add_generation_prompt=True # Añade el token/formato para que el modelo responda
                )
                prompt_applied = True
                logger.debug("Usando plantilla de chat del tokenizador.")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo aplicar chat_template ({e}). Usando formato manual <|role|>.")
                # Formato manual como fallback
                for turn in history:
                    role_tag = "<|user|>" if turn['role'] == 'user' else "<|assistant|>"
                    full_prompt_string += f"{role_tag}\n{turn['content']}\n"
                full_prompt_string += "<|assistant|>\n" # Prompt para la respuesta del asistente

            # --- Tokenización ---
            inputs = tokenizer(full_prompt_string, return_tensors="pt", padding=False, truncation=False) # Sin padding/trunc aquí

            # Mover inputs a CPU (aunque ya deberían estarlo si todo se cargó en CPU)
            inputs = inputs.to(effective_device)

            input_token_length = inputs["input_ids"].shape[-1]
            logger.info(f"📏 Longitud del prompt tokenizado: {input_token_length} tokens")

            # --- Generación ---
            console.print("[bold magenta]Bot:[/bold magenta] ", end="") 
            console.file.flush()  # Flush the output if needed
            # flush=True para asegurar que se muestre "Bot:" inmediatamente
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": max(args.temperature, 1e-3), # Evitar temp=0 que puede dar problemas
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": True if args.temperature > 1e-3 else False, # Samplear solo si temp > 0
                }
                # Cuidado con la longitud total: prompt + max_new_tokens
                # Si excede el límite del modelo, puede dar error o truncar mal.
                # max_model_len = getattr(model.config, 'max_position_embeddings', 2048) # O el límite real
                # if input_token_length + args.max_new_tokens > max_model_len:
                #      logger.warning(f"La longitud combinada ({input_token_length + args.max_new_tokens}) puede exceder el límite del modelo ({max_model_len})")

                output_ids = model.generate(**inputs, **generation_kwargs)

            # --- Decodificación ---
            generated_ids = output_ids[0, input_token_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            console.print(answer)
            print("-" * 20) # Separador

            history.append({"role": "assistant", "content": answer})

            # --- Gestión del Historial ---
            if len(history) > args.max_history * 2:
                history = history[-(args.max_history * 2):]
                logger.debug(f"Historial truncado a {len(history)//2} turnos.")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interacción terminada por el usuario.[/bold yellow]")
    except Exception as e:
        logger.error(f"\n❌ Error inesperado durante el chat: {e}", exc_info=True)
    finally:
        # Limpieza (opcional, Python debería liberar memoria al salir)
        logger.info("Liberando recursos (puede tardar si la RAM es alta)...")
        del model
        del base
        import gc
        gc.collect() # Sugerir recolección de basura
        logger.info("Recursos liberados.")
        log_system_memory(logger)


if __name__ == "__main__":
    main()