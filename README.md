Large Language Model (Using Apache Hadoop)

Author: Pratyay Banerjee
email: pbane8@uic.edu

# Large Language Model Encoder on AWS EMR

## Overview

The goal of this project is to create a **Large Language Model (LLM) Encoder** using parallel distributed computations on **AWS Elastic MapReduce (EMR)**. This project involves training an LLM encoder by processing large datasets efficiently in a distributed environment.

## Project Workflow

1. **Data Sharding and Tokenization**
   - **Sharding**: The input text data is divided into smaller, manageable chunks to facilitate parallel processing across the EMR cluster.
   - **Tokenization**: Each shard is tokenized using [Jtokkit](https://github.com/nocduro/jtokkit), an efficient tokenizer for large-scale text data.

2. **Token Embedding Generation**
   - **MapReduce Job**: In the next MapReduce job, we generate embeddings for the tokens consolidated from the previous step.
   - **Mapper**: Processes the tokens and passes them to the reducer.
   - **Reducer**: Averages the embeddings in the `TokenEmbedding` job to create a unified vector representation for each token.

3. **Cosine Similarity Calculation**
   - **CosineSimilarity Job**: Calculates the cosine similarity of all vector embeddings.
   - **Result**: Generates the top 5 similar words for each word based on the cosine similarity scores.

## Getting Started

#Youtube video Link: https://youtu.be/QWJDm6ITcMg

### Prerequisites

- **AWS Account**: With permissions to create and manage EMR clusters.
- **Java Development Kit (JDK)**: Installed on your local machine.
- **Hadoop MapReduce**: Familiarity with Hadoop and MapReduce paradigms.
- **Jtokkit Library**: Included in the project dependencies.

### Installation

1. **Clone the Repository**

   ```bash
   git clone [https://github.com/yo](https://github.com/prat-bane/LLM-MapReduce)
