[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[mkdocs badge]: <https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat>
[mkdocs]: <https://squidfunk.github.io/mkdocs-material>
[version badge]: <https://img.shields.io/pypi/v/MapIntel.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/MapIntel.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/MapIntel>
[gitter]: <https://gitter.im/MapIntel/community>
[gitter badge]: <https://badges.gitter.im/join%20chat.svg>
[discussions]: <https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel>
[ci]: <https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel/actions?query=workflow>
[ci badge]: <https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel/actions/workflows/ci.yml/badge.svg>
[doc]: <https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel/actions?query=workflow>
[doc badge]: <https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel/actions/workflows/doc.yml/badge.svg?branch=master>

# MapIntel

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![gitter][gitter badge]][gitter] [![discussions][discussions badge]][discussions] |

## Introduction

MapIntel is a system for acquiring intelligence from vast collections of text data by representing each document as a
multidimensional vector that captures its semantics. The system is designed to handle complex Natural Language queries while it
provides Question-Answering functionality. Additionally, it allows for a visual exploration of the corpus. The MapIntel uses a
retriever engine that first finds the closest neighbors to the query embedding and identifies the most relevant documents. It
also leverages the embeddings by projecting them onto two dimensions while preserving the multidimensional landscape, resulting in
a map where semantically related documents form topical clusters which we capture using topic modeling. This map aims to promote a
fast overview of the corpus while allowing a more detailed exploration and interactive information encountering process. MapIntel
can be used to explore many types of corpora.

![MapIntel UI screenshot](./docs/artifacts/ui.png)

## Installation

For user installation, `mapintel` is currently available on the PyPi's repository, and you can install it via `pip`:

```bash
pip install mapintel
```

Development installation requires cloning the repository and then using [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel.git
cd mapintel
pdm install
```

## Configuration

MapIntel aims to be a flexible system that can run with any user provided corpus. In order to achieve this goal, it standardizes
the data and models, while the deployment of all services is expected to be on AWS. An example of how to fully set up a MapIntel
instance can be found at [MapIntel-News](https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/MapIntel-News). After deploying
the required services, a file `.env` should be created at the root of the project with environmental variables that are described
below.

### AWS credentials

The following environmental variable should be included in the `.env` file:

- `AWS_PROFILE_NAME`

The user should have permissions to interact with the services described below.

### Data

An OpenSearch database instance should be deployed in AWS with documents contained in an index called `document`. Each document is
expected to have the `content`, `date`, `embedding`, `embedding2d` and `topic` fields with the following types:

- `content`: text type that contains the main text of the document.
- `date`: `long` type that represents the ordinal format of a date.
- `embedding`: `knn_vector` type that represents the embedding vector of the document.
- `embedding2d`: `float` type that represents the 2D embedding vector of the document.
- `topic`: `keyword` type that assigns a topic label to each document.

The relevant environmental variables are the following:

- `OPENSEARCH_ENDPOINT`: The AWS endpoint of the OpenSearch deployed instance.
- `OPENSEARCH_PORT`: The port of the instance.
- `OPENSEARCH_USERNAME`: The username.
- `OPENSEARCH_PASSWORD`: The password.

### Models

MapIntel uses three models trained on the user provided data. The first is a Haystack retriever model, the second is a model that
reduces the dimensions of the embeddings to 2D, while the third is a generator model used for question-answering. The
corresponding environmental variables are the following:

- `HAYSTACK_RETRIEVER_MODEL`: The value of the parameter `embedding_model` of the Haystack class `EmbeddingRetriever`.
- `SAGEMAKER_DIMENSIONALITY_REDUCTIONER_ENDPOINT`: The SageMaker endpoint of the deployed dimensionality reductioner.
- `SAGEMAKER_GENERATOR_MODEL_ENDPOINT`: The SageMaker endpoint of the deployed generator.

## Usage

To run the application use the following command:

```bash
mapintel
```

Then the server starts and listens to connections at `http://localhost:8080`. You may open the browser and use this URL to
interact with the MapIntel UI.
