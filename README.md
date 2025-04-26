## Pre requisitos
Tener instalado Python. En caso de usar windows tener configurado python en el path

### Consideraciones del script
- Si no funciona usar el comando "python" probar con "python3"
- Si mueves o cambias el nombre del data set, cambiar el parametro "--dataset_path"
- En el ejemplo no esta el parametro "--model_name" pero en caso de usar un modelo diferente agregarlo o en su defecto ir la funcion ***def parse_args*** en train.py y cambiar el valor por defecto de este parametro con la ruta al modelo 

## Script de ejecucion
```shell
python train.py --dataset_path ./data/train.json --bit_precision 4 --batch_size 1 --grad_accum_steps 8 --max_length 128 --save_steps 100 --lora_r 4 --lora_alpha 8
```

## Caso de error no detecta GPU
### Solucion 1 (Solo con hacer esto ya me funciono xd)
Desinstalar torch:
```shell
pip uninstall torch torchvision torchaudio
```

Re instalar la libreria
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Tambien es necesario instalar estas otras librerias:
```shell
pip install transformers datasets peft bitsandbytes torch evaluate rich psutil numpy xformers huggingface_hub accelerate
```

## Script de prueba del modelo 
Este script de chat te levanta el modelo localmente y puedes hablar con el modelo por consola
```shell
python chat.py --base_model ./deepseek-math-7b-instruct --adapter_dir finetuned-math-model --bit_precision 4
```


## Merge model
```shell
python merge.py --base_model ./deepseek-math-7b-instruct --adapter_dir ./finetuned-math-model --output_dir merged_model
```