# How to build custom pytorch on clariden

1. get python from source 
    
    wget [https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz](https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz)
    tar -xf Python-3.11.10.tgz
    cd Python-3.11.10
    ./configure --prefix=$HOME/projects/deps/python311 --enable-shared
    make -j$(nproc)
    make install
    
2. get pytorch from source
    
    git clone --recursive [https://github.com/pytorch/pytorch.git](https://github.com/pytorch/pytorch.git)
    cd pytorch
    git checkout v2.5.1
    git submodule sync
    git submodule update --init --recursive
    
3. get patch and apply patch
    
    curl -o patch_pytorch-2.5.1_uvm-v2.patch "[https://server.example.com/patches/pytorch-uvm/raw/branch/v2.5.1/patch_pytorch-2.5.1_uvm-v2.patch](https://server.example.com/patches/pytorch-uvm/raw/branch/v2.5.1/patch_pytorch-2.5.1_uvm-v2.patch)"
    
    patch -p0 < patch_pytorch-2.5.1_uvm-v2.patch
    
4. Make venv with custom python version
    
    $HOME/projects/deps/python311/bin/python3 -m venv $HOME/projects/deps/pytorch-venv
    
    source $HOME/projects/deps/pytorch-venv/bin/activate
    
5. Set env variables
    
    export _GLIBCXX_USE_CXX11_ABI=1
    
    export CUDA_HOME=/usr/local/cuda
    
    - or: export CUDA_HOME=/user-environment/env/default
    
    export PATH=$HOME/projects/deps/python311/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/projects/deps/python311/lib:$LD_LIBRARY_PATH
    export CPATH=$HOME/projects/deps/python311/include/python3.11:$HOME/projects/deps/pytorch-venv/include/python3.11:$CPATH
    
    export CMAKE_PREFIX_PATH="$HOME/projects/deps/pytorch-venv/lib/python3.11/site-packages:$HOME/projects/deps/python311"
    
6. Install deps
    
    pip3 install --upgrade pip setuptools wheel ninja
    
    pip3 install -r requirements.txt
    
7. Build pytorch
    1. Either (offical code)
        
        $HOME/projects/deps/python311/bin/python3 [setup.py](http://setup.py/) bdist_whee
        
    2. Or: (this is working)
        
        python3 [setup.py](http://setup.py/) develop
        

additionally: unset PYTHONPATH 

## How to make torchvision

1. Clone
    
    git clone [https://github.com/pytorch/vision.git](https://github.com/pytorch/vision.git)
    
    cd vision
    
    git checkout v0.20.1
    
    git submodule sync
    
    git submodule update --init --recursive
    
2. install deps
    1. go to venv
    
    pip3 install --upgrade pip setuptools wheel ninja cmake numpy scapy cython
    
3. install torch with uvm patch (if not there already)
4. set env
    
    export BUILD_VERSION=0.20.1
    
    export CUDA_HOME=/user-environment/env/default
    export CUDA_PATH=/user-environment/env/default
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    export CPLUS_INCLUDE_PATH=/users/fbrunne/projects/deps/python311/include/python3.11:$CPLUS_INCLUDE_PATH
    export CPATH=/users/fbrunne/projects/deps/python311/include/python3.11:$CPATH
    export PATH=/users/fbrunne/projects/deps/python311/bin:$PATH
    export LD_LIBRARY_PATH=/users/fbrunne/projects/deps/python311/lib:$LD_LIBRARY_PATH
    
5. build
    
    python [setup.py](http://setup.py/) bdist_wheel