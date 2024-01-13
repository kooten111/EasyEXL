# EasyEXL Description
A Python script designed to streamline the process of quantizing models to exllamav2 format 

Convert FP16 models from .bin to safetensor (if necessary) and then quantize them with exllama2.

# How to use 
Set the exllamav2 directory and calibration dataset file in settings.json

Then you can just run ``python EasyEXL.py /path/to/model`` optionally you can add the bpw argument to override the settings.json bpw value ``--bpw 4.5``

batch.py will do multiple runs with different bpw values ``python batch.py /path/to/model --bpw "8, 6, 4"``

The quantized model will be in the original models folder.
