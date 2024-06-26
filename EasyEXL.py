#!/usr/bin/env python

import os
import subprocess
import shutil
import json
import time
import argparse

def load_config():
    print("Loading configuration...")
    settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
    with open(settings_path, 'r') as f:
        return json.load(f)

def setup_directories(model_path, config):
    print("Setting up directories...")
    model_path = model_path.rstrip("/")
    return {
        "fp16_model_dir": os.path.abspath(model_path.rstrip("/")),
        "exllama_dir": os.path.abspath(config['exllama_dir']),
        "quant_dir": os.path.join(os.path.abspath(model_path), f"{os.path.basename(model_path)}-{config['bits_per_weight']}bpw-exl2")
    }

def run_conversion_scripts(directories, config):
    print("Convert to safetensors...")
    conversion_script = os.path.join(directories['exllama_dir'], 'util', 'convert_safetensors.py')

    for f in os.listdir(directories['fp16_model_dir']):
        if f.endswith('.bin'):
            basename = os.path.splitext(f)[0]
            safetensor_file = f"{basename}.safetensors"
            safetensor_path = os.path.join(directories['fp16_model_dir'], safetensor_file)

            if not os.path.exists(safetensor_path):
                print(f"Converting {f} to safetensor...")
                try:
                    subprocess.run(['python', conversion_script, os.path.join(directories['fp16_model_dir'], f)])
                except Exception as e:
                    print(f"Error in converting {f}: {e}")

def prepare_quantization_directory(directories):
    print("Preparing quantization directory...")
    if not os.path.exists(directories['quant_dir']):
        os.makedirs(directories['quant_dir'])

def run_quantization(directories, config):
    print("Running quantization...")
    measurement_file = os.path.join(directories['exllama_dir'], f"measurement-{os.path.basename(directories['fp16_model_dir'])}.json")
    measurement_arg = ['-m', measurement_file] if os.path.exists(measurement_file) else []

    convert_py_script = os.path.join(directories['exllama_dir'], 'convert.py')
    if not os.path.isfile(convert_py_script):
        print(f"Conversion script not found: {convert_py_script}")
        return False

    try:
        start_time = time.time()
        result = subprocess.run(['python', convert_py_script, '-i', directories['fp16_model_dir'], '-o', directories['quant_dir'],
                                 '-b', config["bits_per_weight"], '-hb', config["head_bits"], '-l', config["token_length"], '-ml', config["measurement_length"],
                                 '-ra', config["rope_alpha"], '-r', config["dataset_rows"], '-mr', config["measurement_rows"]] + measurement_arg, check=True)
        end_time = time.time()
        duration = end_time - start_time
        if result.returncode == 0:
            print("Quantization successful.")
            print(f"Time taken: {duration // 60} minutes {duration % 60:.2f} seconds")
            return True
        
    except subprocess.CalledProcessError as e:
        print("### ERROR in Quantization ###")
        print(e)
        return False

def cleanup_and_save(directories):
    print("Cleanup temporary files...")
    shutil.rmtree(os.path.join(directories['quant_dir'], 'out_tensor'), ignore_errors=True)

    for file_name in ['cal_data.safetensors', 'job.json', 'input_states.safetensors', 'output_states.safetensors', 'hidden_states.safetensors', 'job_new.json']:
        file_path = os.path.join(directories['quant_dir'], file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    print("Save measurement file...")
    measurement_file = os.path.join(directories['exllama_dir'], f"measurement-{os.path.basename(directories['fp16_model_dir'])}.json")
    if not os.path.exists(measurement_file):
        try:
            shutil.copy(os.path.join(directories['quant_dir'], 'measurement.json'), measurement_file)
        except FileNotFoundError:
            print(f"Failed to find 'measurement.json' in {directories['quant_dir']}.")
    
    print("Copy peripheral files...")
    model_files = [f for f in os.listdir(directories['fp16_model_dir']) if f.endswith('.json') or f.startswith('tokenizer.')]
    for model_file in model_files:
        shutil.copy(os.path.join(directories['fp16_model_dir'], model_file), os.path.join(directories['quant_dir'], model_file))

def main():
    print("#### STARTING SCRIPT ####")
    parser = argparse.ArgumentParser(description='Convert and Quantize fp16 models to Exllama2.')
    parser.add_argument('model_path', type=str, help='Path to FP16 model directory')
    parser.add_argument('--bpw', type=str, help='Bits per weight')
    args = parser.parse_args()

    config = load_config()

    if args.bpw is not None:
        config['bits_per_weight'] = args.bpw

    directories = setup_directories(args.model_path, config)
    
    run_conversion_scripts(directories, config)
    prepare_quantization_directory(directories)
    if run_quantization(directories, config):
        cleanup_and_save(directories)
        print("Finished...")



if __name__ == "__main__":
    main()
