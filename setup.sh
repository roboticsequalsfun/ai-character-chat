#!/bin/bash

sudo apt update
sudo apt install python3.12-venv

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
set -e

ROOT=data

mkdir -p \
    $ROOT/dataset/wavs \
    $ROOT/output/wavs \
    $ROOT/output/text \
    $ROOT/processed/chunks \
    $ROOT/processed/separated \
    $ROOT/processed/transcripts \
    $ROOT/temp/runs \
    $ROOT/raw/audio
