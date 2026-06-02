#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/HW2/data/road_vehicle
cd /root/HW2/data/road_vehicle
kaggle datasets download -d ashfakyeafi/road-vehicle-images-dataset
unzip -o road-vehicle-images-dataset.zip
