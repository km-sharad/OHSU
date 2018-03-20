#!/bin/bash

set -eo pipefail

if (( $# != 1 )); then
   printf "Usage: %s <config.json>\n" "$0" >&2;
   exit 1;
fi;

JSON=$1

function finish {
    # Your cleanup code here
    rm -rf ${TMP}
}
trap finish EXIT

VENV='venv'
OFFSET='OFFSET'
REQ='requirements.txt'
ARCH='venv.zip'
PYTHON='python2.7'

TMP=${VENV}.tmp

virtualenv ${TMP}
virtualenv --relocatable ${TMP} --python=${PYTHON}
source ${TMP}/bin/activate
pip install -r ${REQ}
deactivate

mkdir -p ${OFFSET}
mv ${TMP} ${OFFSET}/${VENV}
cd ${OFFSET}
zip -r ${ARCH} ${VENV}
mv ${ARCH} ..
cd -
