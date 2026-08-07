#!/bin/bash
export PYTHONPATH="$PWD"
bash scripts/run5a_hidden128.sh
bash scripts/run5b_blocks3.sh
bash scripts/run5c_hidden128_blocks3.sh
bash scripts/run5d_hidden96_blocks3.sh
