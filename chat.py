import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
# import evaluate # Eliminado si no se usa
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
import warnings

# Ignorar advertencias específicas si son muy ruidosas (opcional)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Passing `pad_token_id` is deprecated.*")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat interactivo con modelo cuantizado + adaptadores LoRA"
    )
    parser.add_argument("--base_model", type=str,
                        default="./deepseek-math-7b-instruct",
                        help="Ruta o nombre del modelo base (cualificado)")
    parser.add_argument("--adapter_dir", type=str,
                        default="finetuned-math-model",
                        help="Directorio con los adaptadores LoRA (.save_pretrained)")
    parser.add_argument("--bit_precision", type=int, choices=[4, 8], default=4,
                        help="Precisión de cuantización (4 o 8 bits)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Dispositivo: 'auto', 'cpu', 'cuda'. 'auto' es preferido con device_map.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Máximo de tokens a generar por respuesta")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperatura de muestreo (más alto = más creativo, más bajo = más determinista)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling: considera tokens cuya probabilidad acumulada sea >= top_p")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Considera los k tokens más probables en cada paso")
    parser.add_argument("--max_history", type=int, default=5,
                        help="Número máximo de turnos de conversación a mantener en el historial")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(__name__)

def log_gpu_memory(logger, device):
    """Log GPU memory usage"""
    if torch.cuda.is_available() and 'cuda' in str(device):
        try:
            # Intenta obtener memoria para el dispositivo principal si es cuda
            gpu_id = torch.cuda.current_device() if device == torch.device('cuda') else 0 # Asume 0 si device_map usó múltiples
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            logger.info(f"🧠 GPU Memory (Device {gpu_id}): {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except Exception as e:
            logger.warning(f"No se pudo obtener el uso de memoria de la GPU: {e}")
    elif 'cpu' in str(device):
         logger.info("🐌 Modelo cargado en CPU.")

def main():
    args = parse_args()
    logger = setup_logging()

    # --- Configuración de Cuantización ---
    quant_cfg = None
    if args.bit_precision > 0: # Permitir no cuantizar si se pasa 0 o None (no implementado en args pero como idea)
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=(args.bit_precision == 4),
            load_in_8bit=(args.bit_precision == 8),
            bnb_4bit_compute_dtype=torch.float16, # dtype para cómputo interno en 4bit
            bnb_4bit_use_double_quant=True,     # Doble cuantización para ahorrar un poco más
            bnb_4bit_quant_type="nf4"           # Tipo de cuantización (nf4 es popular)
        )
        logger.info(f"⚙️ Configuración de cuantización: {args.bit_precision}-bit ({quant_cfg.bnb_4bit_quant_type if args.bit_precision == 4 else '8-bit'})")

    # --- Carga del Tokenizador ---
    try:
        logger.info(f"🔄 Cargando tokenizador desde: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        # Asegurar que pad_token esté configurado si no existe (importante para generate)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning("⚠️ Tokenizador no tenía pad_token, se estableció a eos_token.")
            else:
                 # Añade un pad token si ni siquiera tiene eos (raro pero posible)
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 logger.warning("⚠️ Tokenizador no tenía pad_token ni eos_token. Se añadió [PAD]. Es posible que necesite reentrenamiento.")

        # Configura el lado del padding para la decodificación correcta (izquierda suele ser mejor para causal LM)
        tokenizer.padding_side = "left"

    except Exception as e:
        logger.error(f"❌ Error cargando el tokenizador: {e}")
        return

    # --- Carga del Modelo Base ---
    try:
        logger.info(f"🔄 Cargando modelo base desde: {args.base_model}")
        # Usar device_map="auto" para distribución automática
        # device_map gestionará en qué dispositivo(s) está el modelo
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_cfg,
            device_map="auto", # Automáticamente usa GPUs/CPU
            trust_remote_code=True,
            # torch_dtype=torch.float16 # Opcional: especificar si no se cuantiza
        )
        # No necesitas .to(device) aquí si usas device_map
        effective_device = base.device # Para saber dónde terminó la mayor parte del modelo
        log_gpu_memory(logger, effective_device)

    except Exception as e:
        logger.error(f"❌ Error cargando el modelo base: {e}")
        return

    # --- Carga de Adaptadores LoRA ---
    try:
        logger.info(f"🔄 Aplicando adaptadores LoRA desde: {args.adapter_dir}")
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        # No necesitas .to(device) aquí tampoco, PeftModel hereda device_map
        model.eval() # ¡Importante! Poner en modo evaluación
        logger.info("✅ Modelo y adaptadores cargados correctamente.")
        log_gpu_memory(logger, effective_device) # Memoria después de cargar adaptadores
    except Exception as e:
        logger.error(f"❌ Error cargando los adaptadores LoRA: {e}")
        return


    # --- Ciclo de Chat Interactivo ---
    logger.info(f"💬 Chat interactivo listo. Modelo: {args.base_model} + {args.adapter_dir}")
    logger.info(f"   Parámetros: temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, max_new={args.max_new_tokens}")
    logger.info("   Escribe 'exit' o 'quit' para salir, o presiona Ctrl+C.")

    history = [] # Lista de diccionarios: {'role': 'user'/'assistant', 'content': '...'}

    try:
        while True:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
            if user_input.strip().lower() in ('exit', 'quit'):
                console.print("[bold yellow]Saliendo...[/bold yellow]")
                break

            # Añadir entrada del usuario al historial
            history.append({"role": "user", "content": user_input})

            prompt_applied = False
            try:
                # Intenta usar la plantilla de chat del tokenizador.
                # add_generation_prompt=True añade el indicador para que el modelo empiece a generar la respuesta.
                # tokenize=False para obtener el string formateado (lo tokenizaremos después)
                full_prompt_string = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                prompt_applied = True
            except Exception as e:
                # Si falla (ej. el tokenizador no tiene plantilla), usar el método manual
                logger.warning(f"⚠️ No se pudo aplicar chat_template ({e}). Volviendo a formato manual.")
                # Reconstruir prompt manualmente (similar a tu método original, pero usando el formato de roles)
                full_prompt_string = ""
                for turn in history:
                    role = turn['role']
                    content = turn['content']
                    # Ajusta estos separadores si tu modelo usa otros!
                    if role == 'user':
                        full_prompt_string += f"<|user|>\n{content}\n"
                    else:
                        full_prompt_string += f"<|assistant|>\n{content}\n"
                # Añade el prompt para la nueva respuesta
                full_prompt_string += "<|assistant|>\n"

            # --- Tokenización ---
            # Tokenizar el prompt completo construido
            # return_tensors="pt" para obtener tensores PyTorch
            # padding=True y truncation=True son importantes si tuvieras un batch, pero aquí es solo 1 secuencia
            # max_length puede ser necesario si el historial crece mucho y quieres truncar la entrada
            # model_max_length = tokenizer.model_max_length # O un valor fijo como 2048, 4096, etc.
            # inputs = tokenizer(full_prompt_string, return_tensors="pt", padding=True, truncation=True, max_length=model_max_length - args.max_new_tokens) # Reserva espacio para la respuesta
            inputs = tokenizer(full_prompt_string, return_tensors="pt")

            # Mover los inputs al dispositivo donde está el modelo (importante si no usaste device_map='auto' o si es CPU)
            # Con device_map='auto', los inputs deben ir a la GPU donde empieza el modelo (usualmente GPU 0)
            # Si el modelo está en CPU, debe ir a CPU.
            input_device = model.device # PeftModel debería tener el device correcto
            inputs = inputs.to(input_device)

            input_token_length = inputs["input_ids"].shape[-1]
            logger.info(f"📏 Longitud del prompt tokenizado: {input_token_length} tokens")

            # --- Generación ---
            console.print("[bold magenta]Bot:[/bold magenta] ", end="")
            with torch.no_grad(): # Asegura que no se calculen gradientes
                # Parámetros de generación
                generation_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "do_sample": True, # Necesario para temperature, top_p, top_k
                    # "num_return_sequences": 1 # Por defecto es 1
                }
                output_ids = model.generate(**inputs, **generation_kwargs)

            # --- Decodificación y Extracción (Método Robusto) ---
            # Decodificar SOLO los tokens generados, no el prompt + tokens
            # output_ids contiene [prompt_tokens + generated_tokens]
            # Seleccionamos solo la parte generada
            generated_ids = output_ids[0, input_token_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Imprimir la respuesta
            console.print(answer)
            print("-" * 20) # Separador visual

            # Añadir respuesta del bot al historial
            history.append({"role": "assistant", "content": answer})

            # --- Gestión del Historial (Truncamiento Simple) ---
            # Mantener solo los últimos N turnos (cada turno es user + assistant)
            if len(history) > args.max_history * 2: # Multiplicado por 2 porque cada turno tiene 2 entradas
                 # Mantiene los últimos N*2 mensajes (N preguntas + N respuestas)
                 history = history[-(args.max_history * 2):]
                 logger.debug(f"Historial truncado a {args.max_history} turnos.")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interacción terminada por el usuario.[/bold yellow]")
    except Exception as e:
        logger.error(f"\n❌ Error inesperado durante el chat: {e}", exc_info=True) # Muestra traceback
    finally:
        # Opcional: Limpiar memoria GPU si es necesario
        del model
        del base
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
        logger.info("Recursos liberados.")

if __name__ == "__main__":
    main()