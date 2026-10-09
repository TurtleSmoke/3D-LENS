#!/bin/bash
# Step 7. Write the train_all.txt image list read by the training code, for a real or a synthetic dataset directory.
#   bash generation/build_lists.sh AG-ReID     <dir>   # lists <dir>/bounding_box_train
#   bash generation/build_lists.sh AG-ReID.v2  <dir>   # lists <dir>/AG-ReID.v2/train_all (paths relative to <dir>/AG-ReID.v2)
#   bash generation/build_lists.sh MOO         <dir>   # lists <dir>/output_screenshots_png
set -e
dataset=$1
dir=$2

list() { (cd "$1" && find "$2" -name "*.jpg" -o -name "*.png" | sort) > "$dir/train_all.txt"; }

case $dataset in
    AG-ReID) list "$dir" bounding_box_train ;;
    AG-ReID.v2) list "$dir/AG-ReID.v2" train_all ;;
    MOO) list "$dir" output_screenshots_png ;;
    *) echo "Unknown dataset $dataset" && exit 1 ;;
esac
echo "$(wc -l < "$dir/train_all.txt") images listed in $dir/train_all.txt"
