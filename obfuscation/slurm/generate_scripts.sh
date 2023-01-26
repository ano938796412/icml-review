#!/bin/bash

# generate batch script on slurm to run with ./run_batch.sh
# parameters -m models -a attacks -o options -l log_name -d dir_name -c start_count -r repeat
# options (key=value) are passed into --cfg-options in run.py

# defaults
log_name=$(date +"%y-%m-%d-%H-%M-%S")
dir_name="$log_name"
start_count=0
repeat=1

# "getopts..." end with :
while getopts "m:a:o:l:d:c:r:" opt; do
  case $opt in
  m) models+=("$OPTARG") ;;
  a) attack+=("$OPTARG") ;;
  o) options+=("$OPTARG") ;;
  l) log_name=$OPTARG ;;
  d) dir_name=$OPTARG ;;
  c) start_count=$OPTARG ;;
  r) repeat=$OPTARG ;;
  *)
    echo "usage: $0 [-m] [-a] [-o] [-l] [-d] [-c] [-r] " 1>&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

echo models: "${models[@]}"
echo attack: "${attack[@]}"
echo options: "${options[@]}"
echo log_name: "$log_name"
echo dir_name: "$dir_name"
echo start_count: "$start_count"
echo repeat: "$repeat"

# reset directory if $start_count equals 0
if [ "$start_count" -eq 0 ]; then
  rm -rf "$dir_name"
  mkdir "$dir_name"
fi

cd "$dir_name" || exit

count="$start_count"

for rep in $(seq "$repeat"); do
  plant=$RANDOM

  for a in "${attack[@]}"; do
    for m in "${models[@]}"; do
      ((count++))

      if [ ${#options[@]} -ge 1 ]; then
        cmd="python ./src/run.py --config configs/${a}_${m}.py --log ${log_name}_repeat_${rep} --cfg-options ${options[*]} seed=${plant} shuffle=True"
      else
        cmd="python ./src/run.py --config configs/${a}_${m}.py --log ${log_name}_repeat_${rep} --cfg-options seed=${plant} shuffle=True"
      fi

      echo "$cmd" >"$count.sh"
    done
  done
done

cp ../run_batch.sh .
sed -i -e "s/^\#SBATCH --array=1.*/\#SBATCH --array=1-${count}/g" \
  -e "s/^\#SBATCH --job-name=.*/\#SBATCH --job-name=${dir_name}/g" run_batch.sh
