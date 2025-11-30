#!/bin/bash

# Remote details
REMOTE_USER="ym02ylba"
REMOTE_HOST="lms41-22.e-technik.uni-erlangen.de"
REMOTE_DIR="/home/ym02ylba/repos/photonics/experiments/checkpoint_084"

# Local details
LOCAL_DIR="/home/hpc/iwnt/iwnt153h/photonics/experiments/test_experiment/checkpoints/checkpoint_084"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Sync command using rsync
# -a: archive mode (preserves permissions, times, etc.)
# -v: verbose
# -z: compress during transfer
# -P: show progress

# The trailing slash on REMOTE_DIR ensures we sync the *contents* of the folder
# echo "Syncing from ${REMOTE_HOST}:${REMOTE_DIR} to ${LOCAL_DIR}..."
# rsync -avzP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" "$LOCAL_DIR/"

# Sync from local to remote
echo "Syncing from ${LOCAL_DIR} to ${REMOTE_HOST}:${REMOTE_DIR}..."
rsync -avzP "$LOCAL_DIR/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
