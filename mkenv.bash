#!/bin/bash

# 创建输出目录
mkdir -p dependency_files

# 生成conda环境的完整依赖
echo "Generating conda packages list..."
conda list --explicit > dependency_files/conda-packages.txt

# 生成requirements.txt
echo "Generating requirements.txt..."
pip freeze > dependency_files/requirements.txt

# 生成pyproject.toml
echo "Generating pyproject.toml..."
cat > dependency_files/pyproject.toml << EOL
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "your-project-name"
version = "0.1.0"
description = "Your project description"
requires-python = "$(python -V | cut -d' ' -f2)"
dependencies = [
$(pip freeze | sed 's/^/    "/' | sed 's/$/",/')
]

[tool.setuptools]
packages = ["your_package_name"]
EOL

# 生成setup.py
echo "Generating setup.py..."
cat > dependency_files/setup.py << EOL
from setuptools import setup, find_packages

setup(
    name="your-project-name",
    version="0.1.0",
    packages=find_packages(),
    python_requires="$(python -V | cut -d' ' -f2)",
    install_requires=[
$(pip freeze | sed 's/^/        "/' | sed 's/$/",/')
    ],
)
EOL

echo "Done! Files generated in dependency_files directory:"
ls -l dependency_files/