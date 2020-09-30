#!/bin/bash

# Example usage: docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh

cd /workspace/TRTorch/py

export CXX=g++
export LD_LIBRARY_PATH=/root/.cache/bazel/_bazel_root/dbd0cda766b91a08d9f8b4d01d528393/external/tensorrt/targets/x86_64-linux-gnu/lib/:$LD_LIBRARY_PATH


build_torch() {
    PYTHON=$1
    PY_VERSION=$2

    git clone --recursive https://github.com/pytorch/pytorch.git /pytorch
    pushd /pytorch
    git checkout v1.6.0
    git submodule update --init --recursive
    $PYTHON -m pip install -r requirements.txt
    $PYTHON setup.py install
    popd
}

build_whl() {

    PYTHON=$1
    PY_VERSION=$2

    $PYTHON -m pip install --upgrade pip
    $PYTHON -m pip install -r requirements.txt
    $PYTHON -m pip install auditwheel

    $PYTHON setup.py bdist_wheel --use-cxx11-abi
    PY_PATHS="$(find /opt/_internal -name csrc | grep $PY_VERSION)/../../../"

    WHL_FILE=$(find dist/ -name trtorch*.whl | grep "3${PY_VERSION}")
    AUDITWHEEL=$(find / -name auditwheel | grep ${PY_VERSION} | grep -v "site-packages")

    echo "LD_LIBRARY_PATH=$PY_PATHS/lib:$LD_LIBRARY_PATH auditwheel -v repair --plat manylinux1_x86_64 $WHL_FILE"
    LD_LIBRARY_PATH=$PY_PATHS/lib:$LD_LIBRARY_PATH $AUDITWHEEL -v repair --plat manylinux1_x86_64 $WHL_FILE 
}


for PYTHON in /opt/python/cp3{6,7,8}*/bin/python
do
    PY_VERSION=$(echo $PYTHON | awk -F"cp3|-" '{print $2}')
    build_torch $PYTHON $PY_VERSION
    build_whl $PYTHON $PY_VERSION
    rm -rf /pytorch
done
