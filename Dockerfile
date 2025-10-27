# Use PyTorch with CUDA and cuDNN
# Use miniforge, make sure it has the latest GLIBC (>=2.32)
FROM condaforge/miniforge3:latest
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libc-bin && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH="/opt/conda/lib"
RUN CONDA_OVERRIDE_CUDA=12.9 mamba install python=3.12 pytorch=2.7.1=*cuda129* \
    cuda-toolkit=12.9 \
    cuda-nvcc=12.9 \
    cuda-cudart-dev=12.9 \
    transformer-engine-torch==2.5.0 \
    cudnn -y
RUN pip install -U pip && pip install ninja packaging lightning --no-cache-dir && \
    pip install triton vtx && pip install -U flash-attn==2.7.4.post1 --no-build-isolation
    # flash attention 2.7.4.post1 for transformer engine compatibility
ENV CUDA_HOME=/opt/conda/targets/x86_64-linux
ENV CUDACXX=$CUDA_HOME/bin/nvcc
ENV CPATH=/opt/conda/include:
ENV CPATH=$CUDA_HOME/include:$CPATH
ENV LIBRARY_PATH="/opt/conda/lib"
ENV LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/libcublas:$CUDA_HOME/libcusparse:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/libcublas:$CUDA_HOME/libcusparse:$LD_LIBRARY_PATH
RUN echo "/opt/conda/lib" > /etc/ld.so.conf.d/conda-env.conf && echo "${CUDA_HOME}/lib" >> /etc/ld.so.conf.d/conda-env.conf \
    && echo "${CUDA_HOME}/libcublas" >> /etc/ld.so.conf.d/conda-env.conf \
    && echo "${CUDA_HOME}/libcusparse" >> /etc/ld.so.conf.d/conda-env.conf \
    && ldconfig
RUN ldconfig -p | grep libnvrtc || echo "libnvrtc not found"
RUN pip install --no-cache-dir  jsonargparse[signatures] tokenizers sentencepiece wandb torchmetrics \
                tensorboard zstandard pandas pyarrow huggingface_hub numpy matplotlib scikit-learn scipy tqdm notebook \
                fsspec h5py hydra-core nvitop omegaconf seaborn timm rich biopython lmdb \
                datasets tomli>=1.1.0 transformers einops braceexpand \
                smart_open opt_einsum cbor2 isort pytest torchdata esm lm-eval==0.4.1 torch==2.7.1 \
    && pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/test/cu129 \
    && pip install "nemo_toolkit[all]" && pip install flash-linear-attention 
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && \
    pip install . && \
    cd .. 
RUN  git clone https://github.com/state-spaces/mamba.git && \
     cd mamba && \
     MAX_JOBS=2 pip install . && \
     cd .. 
# RUN mamba install -y gxx_linux-64=12.* gcc_linux-64=12.*
# RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
#     cd flash-attention/ && git checkout 27f501d && cd hopper/ && MAX_JOBS=1 python setup.py install && \
#     python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
#     mkdir -p $python_path/flash_attn_3 && \
#     wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py 
RUN mamba clean --all -y && conda clean -a -y && apt-get clean && \
    pip cache purge && \
    rm -rf /root/.cache/pip && \
    rm -rf /causal-conv1d /mamba 

# check torch version
RUN python -c "import torch; print(torch.__version__); print('CUDA used to build torch (torch.version.cuda):', torch.version.cuda)"
