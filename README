# Telco Customer Churn Prediction — PySpark MLlib Pipeline

End-to-end machine learning pipeline for binary churn classification, built with **PySpark MLlib** on a **Hadoop/YARN** distributed cluster. Trains and compares three classifiers, evaluates on AUC-ROC, F1, and Accuracy, and persists models and predictions to HDFS.
 
---

## Stack
 
- **PySpark** — distributed data processing and ML
- **PySpark MLlib** — Pipeline, feature transformers, classifiers, evaluators
- **Hadoop / HDFS** — distributed storage (input CSV + model output)
- **YARN** — cluster resource management
- **Python 3.x**
 
---


## Dataset
 
[Telco Customer Churn — IBM Sample Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
 
- ~7,000 customer records
- 21 features: demographics, account info, services subscribed
- Target: `Churn` (Yes/No)
- Class imbalance: ~73% No / ~27% Yes
 
Place the CSV at: `hdfs://localhost:9000/churn/input/Telco-Customer-Churn.csv`
 
---
 
## Setup & Run
 
### Prerequisites
 
- Java 8+
- Hadoop 3.x running locally (NameNode + DataNode)
- Apache Spark 3.x with PySpark
- Python 3.8+
 
### Install Python dependencies
 
```bash
pip install pyspark
```
 
### Start Hadoop
 
```bash
start-dfs.sh
start-yarn.sh
```
 
### Upload dataset to HDFS
 
```bash
hdfs dfs -mkdir -p /churn/input
hdfs dfs -put Telco-Customer-Churn.csv /churn/input/
```
 
### Run the pipeline
 
```bash
spark-submit churn_train.py
```
 
---
 