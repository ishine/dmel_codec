from setuptools import setup, find_packages

setup(
    name="dmel_codec",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # core
        "numpy>=2.0.0",
        "torch>=2.5.0",
        "torchaudio>=2.5.0",
        "einops==0.8.0",
        "huggingface_hub==0.26.2",
        "hydra-core==1.3.2",
        "librosa==0.10.2.post1",
        "lightning==2.4.0",
        "matplotlib>=3.9.0",
        "omegaconf==2.3.0",
        "rich==13.9.4",
        "rootutils==1.0.7",
        "scipy>=1.14.0",
        "soundfile==0.12.1",
        "vector-quantize-pytorch>=1.20.9",
        
        # optional
        "tensorboard>=2.0.0",
        "wandb>=0.18.0",
    ],
    python_requires=">=3.10",
)