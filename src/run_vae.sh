#!/bin/sh

THIS_DIR=$(cd $(dirname $0); pwd)

cd ${THIS_DIR}
time python -m sandbox.vae.main $@
