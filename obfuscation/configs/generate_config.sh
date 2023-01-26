#!/bin/bash

# generate configs given model name(s) as argument
# (corresponding model config in ./models should exist)
# ./generate_config.sh faster_rcnn yolo_v3 retinanet ssd_512 cascade_rcnn

template_model="faster_rcnn"
template_dir="templates"

for model in "$@"; do
  config_path="./models/$model.py"
  if [ -e $config_path ]; then # check model cfg exists
    for tm in ./$template_dir/*.py; do
      tm_name=$(basename $tm)
      save_name=${tm_name/$template_model/$model}             # replace string in fname
      sed -e "s/$template_model/$model/g" "$tm" >"$save_name" # and in contents

      echo "created $save_name using $tm"
    done
  else
    echo "$config_path does not exist"
  fi
done
