#!/bin/bash

set -euo pipefail

if (( $# != 3 )); then
   printf "Usage: %s <config.json> <num-executors> <prog.py>\n" "$0" >&2;
   exit 1;
fi;

JSON=$1
NEXE=$2
PROG=$3

VENV='venv'
OFFSET='OFFSET'
ARCH='venv.zip'

PY=./venv/bin/python
PYTHON=./${OFFSET}/${PY}

PYSPARK_PYTHON=${PYTHON} \
              spark-submit \
              --conf spark.yarn.appMasterEnv.PYSPARK_Python=${PYTHON} \
              --master yarn \
              --archives ${ARCH}#${OFFSET} \
              --driver-memory 5g \
              --executor-memory 7g \
              --num-executors ${NEXE} \
              ${PROG}
