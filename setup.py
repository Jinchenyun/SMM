from setuptools import setup, find_packages

setup(
    name="SMM",
    version="0.1.0",
    author="Jin Chenyun",
    author_email="1693277228@qq.com",
    description="Support Matrix Machine Classifier",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smm-classifier",
    packages=find_packages(exclude=["test*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.26.4',
        'cvxpy>=1.6.6',
        'scikit-learn>=1.5.2'
    ],
)
