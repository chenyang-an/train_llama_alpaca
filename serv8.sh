# conda create -n llm python=3.10 -y
# conda deactivate && conda activate llm
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install vllm


conda install -c nvidia cuda-nvcc
pip install transformers==4.34.0
pip install "fschat[model_worker,train]"

