#!/bin/bash


INPUT_BASE_PATH="./checkpoints-40m-qk-layernorm"  
STEPS="0 1 2 4 8 16 32 64 128 256 512 1000 2000 3000 4000 4091"     
OUTPUT_FILE="./hf-checkpoints-40m-qk-layernorm/step"    
YML_FILE="./models/pythia-40m-qk-layernorm.yml"   
PRECISION="fp32"                   

for step in $STEPS; do
    python ./tools/ckpts/convert_neox_to_hf.py \
    --input_dir "${INPUT_BASE_PATH}/global_step${step}" \
    --config_file "${YML_FILE}" \
    --output_dir "${OUTPUT_FILE}${step}" \
    --precision "${PRECISION}" 
done