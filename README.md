Large Language Model (Using Apache Hadoop)

Author: Pratyay Banerjee
email: pbane8@uic.edu

Large Language Model Encoder on AWS EMR
Overview
The goal of this project is to create a Large Language Model (LLM) Encoder using parallel distributed computations on AWS Elastic MapReduce (EMR). This project involves training an LLM encoder by processing large datasets efficiently in a distributed environment.

Project Workflow
Data Sharding and Tokenization

Sharding: The input text data is divided into smaller, manageable chunks to facilitate parallel processing across the EMR cluster.
Tokenization: Each shard is tokenized using Jtokkit, an efficient tokenizer for large-scale text data.
Token Embedding Generation

MapReduce Job: In the next MapReduce job, we generate embeddings for the tokens consolidated from the previous step.
Mapper: Processes the tokens and passes them to the reducer.
Reducer: Averages the embeddings in the TokenEmbedding job to create a unified vector representation for each token.
Cosine Similarity Calculation

CosineSimilarity Job: Calculates the cosine similarity of all vector embeddings.
Result: Generates the top 5 similar words for each word based on the cosine similarity scores.
Getting Started
Prerequisites
AWS Account: With permissions to create and manage EMR clusters.
Java Development Kit (JDK): Installed on your local machine.
Hadoop MapReduce: Familiarity with Hadoop and MapReduce paradigms.
Jtokkit Library: Included in the project dependencies.
Installation
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/llm-encoder-aws-emr.git
cd llm-encoder-aws-emr
Install Dependencies

Ensure that all required libraries and dependencies are installed.
Configure your AWS credentials for EMR access.
Usage
Data Preparation
Input Data: Place your input text data in an accessible location, such as an S3 bucket if using AWS EMR.
Configuration: Update the configuration files with the correct paths to your input and output data.
Running the Jobs
Data Sharding and Tokenization

bash
Copy code
hadoop jar llm-encoder.jar com.yourpackage.TokenizationJob -D input=/path/to/input -D output=/path/to/output
Token Embedding Generation

bash
Copy code
hadoop jar llm-encoder.jar com.yourpackage.TokenEmbeddingJob -D input=/path/to/tokenized/output -D output=/path/to/embeddings/output
Cosine Similarity Calculation

bash
Copy code
hadoop jar llm-encoder.jar com.yourpackage.CosineSimilarityJob -D input=/path/to/embeddings/output -D output=/path/to/cosine/output
Viewing the Results
The final output will contain the top 5 similar words for each word based on cosine similarity.
Check the output directory specified in the last job for the results.
Project Structure
css
Copy code
llm-encoder-aws-emr/
├── src/
│   ├── com/yourpackage/
│   │   ├── TokenizationJob.java
│   │   ├── TokenEmbeddingJob.java
│   │   └── CosineSimilarityJob.java
├── data/
│   └── input/
├── output/
├── README.md
└── pom.xml
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Jtokkit for providing an efficient tokenization tool.
AWS EMR for scalable and distributed computing resources.





