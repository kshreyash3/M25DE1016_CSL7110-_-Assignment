# CSL7110 Assignment 1: MapReduce and Apache Spark

[cite_start]This repository contains the code for Assignment 1, focusing on big data processing using Apache Hadoop and Apache Spark[cite: 1, 12, 96]. [cite_start]The project processes text data from the Project Gutenberg dataset[cite: 63].

##  Project Overview

The assignment is divided into two main parts:
1. [cite_start]**MapReduce (Java)**: Implementing the WordCount algorithm[cite: 7, 20].
2. [cite_start]**Apache Spark (PySpark)**: Performing text analytics, TF-IDF scoring, and network generation[cite: 7, 120, 136].

---

## Scripts and Tasks

### Part 1: Hadoop MapReduce
* [cite_start]**`WordCount.java`**: A MapReduce program written in Java that counts the frequency of words in a given text file, utilizing standard Hadoop `Mapper` and `Reducer` classes and ignoring punctuation[cite: 51, 56, 58].

### Part 2: Apache Spark
* **`task10_metadata_analysis.py`**: Extracts metadata (Title, Release Date, Language, Character Encoding) from raw text files using regular expressions. [cite_start]It outputs the number of books released per year, the most common language, and the average title length[cite: 107, 108, 109, 110, 111, 113, 114, 115].
* [cite_start]**`task11_tfidf_similarity.py`**: Cleans the text, tokenizes words, removes stop words, and calculates TF-IDF scores for the documents[cite: 123, 124, 128]. [cite_start]It uses these scores to compute the Cosine Similarity and find the top 5 books most similar to a target book[cite: 131, 132].
* [cite_start]**`task12_author_network.py`**: Constructs a network representing author influence based on publication dates within a 5-year window[cite: 138, 144]. [cite_start]It calculates and outputs the top 5 authors with the highest in-degree and out-degree[cite: 149, 151].

---

##  How to Run

**For MapReduce (Java):**
```bash
# Execute the compiled jar on the Hadoop cluster
hadoop jar WordCount.jar output/


# Run the Python scripts directly using spark-submit
spark-submit task10_metadata_analysis.py
spark-submit task11_tfidf_similarity.py
spark-submit task12_author_network.py

# View the output
hadoop fs -getmerge output/ output.txt
cat output.txt
