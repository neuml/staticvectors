# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    # Remove GitHub dark mode images
    DESCRIPTION = "".join([line for line in f if "gh-dark-mode-only" not in line])

setup(
    name="staticvectors",
    version="0.1.0",
    author="NeuML",
    description="Work with static vector models",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neuml/staticvectors",
    project_urls={
        "Documentation": "https://github.com/neuml/staticvectors",
        "Issue Tracker": "https://github.com/neuml/staticvectors/issues",
        "Source Code": "https://github.com/neuml/staticvectors",
    },
    license="Apache 2.0: http://www.apache.org/licenses/LICENSE-2.0",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    keywords="search embedding machine-learning nlp",
    python_requires=">=3.9",
    install_requires=["huggingface-hub>=0.19.0", "numpy>=1.18.4", "safetensors>=0.4.5", "tqdm>=4.48.0"],
    extras_require={"train": ["fasttext-wheel>=0.9.2", "nanopq>=0.2.1"]},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Utilities",
    ],
)
