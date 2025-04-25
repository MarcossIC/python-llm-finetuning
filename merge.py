import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rich.logging import RichHandler
from rich.console import Console
import warnings

# Ignorar advertencias ruidosas opcionales
warnings.filterwarnings("ignore", category=UserWarning, message=".*Passing `pad_token_id` is deprecated.*")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fusiona un modelo base con adaptadores LoRA/QLoRA y guarda el resultado"
    )
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="Ruta o nombre del modelo base (pre-entrenado)"
    )
    parser.add_argument(
        "--adapter_dir", type=str, required=True,
        help="Directorio con los adaptadores LoRA/QLoRA (.save_pretrained)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="merged_model",
        help="Directorio donde se guardará el modelo fusionado"
    )
    parser.add_argument(
        "--save_tokenizer", action="store_true",
        help="Guardar también el tokenizer en el output_dir"
    )
    parser.add_argument(
        "--bit_precision", type=int, choices=[0,4,8], default=0,
        help="Cuantización: 4, 8 bits o 0 para sin cuantizar"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Dispositivo: 'auto', 'cpu', o 'cuda'"
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    logger = setup_logging()

    # Configuración de cuantización si aplica
    quant_cfg = None
    if args.bit_precision in (4, 8):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=(args.bit_precision == 4),
            load_in_8bit=(args.bit_precision == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        logger.info(f"⚙️ Cuantización: {args.bit_precision}-bit configurada")

    # Cargar modelo base
    try:
        logger.info(f"🔄 Cargando modelo base desde: {args.base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quant_cfg,
            device_map="auto" if args.device == "auto" else None,
            torch_dtype=torch.float16 if quant_cfg and args.bit_precision==4 else None,
            trust_remote_code=True
        )
        logger.info("✅ Modelo base cargado correctamente.")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo base: {e}")
        return

    # Cargar adaptadores LoRA/QLoRA
    try:
        logger.info(f"🔄 Aplicando adaptadores desde: {args.adapter_dir}")
        peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
        peft_model.eval()
        logger.info("✅ Adaptadores cargados correctamente.")
    except Exception as e:
        logger.error(f"❌ Error cargando adaptadores: {e}")
        return

    # Fusionar y descargar adaptadores
    try:
        logger.info("🔀 Fusionando adaptadores en el modelo base...")
        merged = peft_model.merge_and_unload()
        logger.info("✅ Fusión completada.")
    except Exception as e:
        logger.error(f"❌ Error durante la fusión: {e}")
        return

    # Guardar modelo fusionado
    try:
        logger.info(f"💾 Guardando modelo fusionado en: {args.output_dir}")
        merged.save_pretrained(args.output_dir)
        logger.info("✅ Modelo fusionado guardado correctamente.")
    except Exception as e:
        logger.error(f"❌ Error al guardar el modelo fusionado: {e}")
        return

    # Guardar tokenizer si se solicita
    if args.save_tokenizer:
        try:
            logger.info("💾 Guardando tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            tokenizer.save_pretrained(args.output_dir)
            logger.info("✅ Tokenizer guardado correctamente.")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar el tokenizer: {e}")

    logger.info("🎉 Proceso completado. Ahora tienes un modelo fusionado listo para usar.")


if __name__ == "__main__":
    main()
