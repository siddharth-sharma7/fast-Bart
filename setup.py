from setuptools import setup, find_packages

setup(
    name='fastBart',
    version='1.0',
    packages=find_packages(),
    install_requires=[
    'torch>=1.7.0,!=1.8.0', # excludes torch v1.8.0
    'onnx',
    'onnxruntime==1.7.0',
    'transformers==4.11.3',
    'progress>=1.5',
    'sentencepiece',
    'psutil',
    'numpy',
    'matplotlib',
    ]
)
