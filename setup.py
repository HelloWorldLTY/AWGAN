from setuptools import setup, find_packages

setup(
    name="awgan",
    packages=find_packages(),
    version='0.0.6',
    description="code for model AWGAN",
    author="tianyu liu",
    author_email='tianyul4@illinois.edu',
    url="https://github.com/HelloWorldLTY/AWGAN",
    keywords=['single-cell', 'rna', 'GAN', 'deep learning'],
    classifiers=[],
    install_requires=[
        'torch',
        'numpy',
        'scanpy',
        'numba',
        'scikit-misc',
        'graphtools'
    ]
)
