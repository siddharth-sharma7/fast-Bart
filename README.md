# fast-Bart

### Reduction of BART model size by 3X, and boost in inference speed up to 3X
  BART implementation of the fastT5 library (https://github.com/Ki6an/fastT5)
  
  **Pytorch model -> ONNX model -> Quantized ONNX model**

---
## Install

Install using requirements.txt file

```python
git clone https://github.com/siddharth-sharma7/fast-Bart
cd fast-Bart
pip install -r requirements.txt
```
---
## Usage

The `export_and_get_onnx_model()` method exports the given pretrained Bart model to onnx, quantizes it and runs it on the onnxruntime with default settings. The returned model from this method supports the `generate()` method of huggingface.

> If you don't wish to quantize the model then use `quantized=False` in the method.

```python
from fastBart import export_and_get_onnx_model
from transformers import AutoTokenizer

model_name = 'facebook/bart-base'
model = export_and_get_onnx_model(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input = "This is a very long sentence and needs to be summarized."
token = tokenizer(input, return_tensors='pt')

tokens = model.generate(input_ids=token['input_ids'],
               attention_mask=token['attention_mask'],
               num_beams=3)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
print(output)
```

> to run the already exported model use `get_onnx_model()`

you can customize the whole pipeline as shown in the below code example:

```python
from fastBart import (OnnxBart, get_onnx_runtime_sessions,
                    generate_onnx_representation, quantize)
from transformers import AutoTokenizer

model_or_model_path = 'facebook/bart-base'

# Step 1. convert huggingfaces bart model to onnx
onnx_model_paths = generate_onnx_representation(model_or_model_path)

# Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
# The process is slow for the decoder and init-decoder onnx files (can take up to 15 mins)
quant_model_paths = quantize(onnx_model_paths)

# step 3. setup onnx runtime
model_sessions = get_onnx_runtime_sessions(quant_model_paths)

# step 4. get the onnx model
model = OnnxBart(model_or_model_path, model_sessions)

                      ...
```
##### custom output paths 
By default, fastBart creates a `models-bart` folder in the current directory and stores all the models. You can provide a custom path for a folder to store the exported models. And to run already `exported models` that are stored in a custom folder path: use `get_onnx_model(onnx_models_path="/path/to/custom/folder/")`

```python
from fastBart import export_and_get_onnx_model, get_onnx_model

model_name = "facebook/bart-base"
custom_output_path = "/path/to/custom/folder/"

# 1. stores models to custom_output_path
model = export_and_get_onnx_model(model_name, custom_output_path)

# 2. run already exported models that are stored in custom path
# model = get_onnx_model(model_name, custom_output_path)
```
## Functionalities

- Export any pretrained Bart model to ONNX easily.
- The exported model supports beam search and greedy search and more via `generate()` method.
- Reduce the model size by `3X` using quantization.
- Up to `3X` speedup compared to PyTorch execution for greedy search and `2-3X` for beam search.

