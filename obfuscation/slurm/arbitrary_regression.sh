#!/bin/bash

# generate arbitrary bbox scripts to regress success rate against bbox lengths and boundary distances

bbox_lengths=(200 100 50 10)
boundary_distances=(10 50 100 200)

attack_samples=100
repeat=2

itrs=200

echo bbox_lengths: "${bbox_lengths[@]}"
echo boundary_distances: "${boundary_distances[@]}"
echo attack_samples: "${attack_samples}"
echo repeat: "${repeat}"
echo itrs: "${itrs}"
echo

count=0

for len in "${bbox_lengths[@]}"; do
  for dist in "${boundary_distances[@]}"; do
    echo

    ./generate_scripts.sh -m faster_rcnn -m yolo_v3 -m retinanet -m ssd_512 -m cascade_rcnn -a vanish_arbitrary -a mislabel_arbitrary -a untarget_arbitrary -d arbitrary -l bbox_"${len}"_dist_"${dist}" -o perturb_kwargs.arbitrary_bbox_length="${len}" -o perturb_kwargs.boundary_distance="${dist}" -o result_dir="./data/arbitrary/results" -o dataset_dir="./data/arbitrary/datasets" -o log_dir="./data/arbitrary/logs" -o img_dir="./data/arbitrary/images" -o attack_samples="${attack_samples}" -r "${repeat}" -o itrs="${itrs}" -c "${count}"

    ((count += 15 * repeat)) # 5 models and 3 attacks
  done
done

echo
echo count: "$count"
