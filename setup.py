from setuptools import setup, find_packages

setup(
    name="bugbusterai",
    version="0.1.0",
    description="Advanced AI for Code Bug Detection and Localization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0", 
        "astor>=0.8.1",
        "networkx>=2.6.3",
        "tqdm>=4.62.3",
        "python-igraph>=0.9.11",
        "pytest>=7.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Bug Tracking",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)