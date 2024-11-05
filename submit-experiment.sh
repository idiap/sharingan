#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

## Create a timestamp-based folder for the experiment
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
EXPERIMENT_PATH="experiments/$DATE/$TIME"
mkdir -p "$EXPERIMENT_PATH"

## Copy a code snapshot
cp -r "src" "main.py" "$EXPERIMENT_PATH"

## Switch to the experiment folder
cd "$EXPERIMENT_PATH"

## Launch experiment
python main.py --config-name "config_gf" # update accordingly