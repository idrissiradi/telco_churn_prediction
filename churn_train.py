from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, trim, when

##  Chargement + nettoyage
spark = (
    SparkSession.builder.appName("Churn-ML-Pipeline")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .master("local[*]")
    .getOrCreate()
)

path = "/churn/input/Telco-Customer-Churn.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(path)

print("Schema:")
df.printSchema()
print("Rows:", df.count())

if "TotalCharges" in df.columns:
    df = df.withColumn(
        "TotalCharges",
        when(trim(col("TotalCharges")) == "", lit(None))
        .otherwise(col("TotalCharges"))
        .cast("double"),
    )

df = df.dropna(subset=["Churn"])
df.select("Churn").groupBy("Churn").count().show()

## Feature Engineering
id_cols = [c for c in ["customerID", "CustomerID"] if c in df.columns]
target_col = "Churn"
numeric_cols = []
categorical_cols = []

for c, t in df.dtypes:
    if c in id_cols or c == target_col:
        continue
    if t in ("int", "bigint", "double", "float"):
        numeric_cols.append(c)
    else:
        categorical_cols.append(c)

print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

df = df.withColumn("label", when(col("Churn") == "Yes", 1).otherwise(0))
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]
encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical_cols
]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[f"{c}_imputed" for c in numeric_cols],
    strategy="median",
)
df = imputer.fit(df).transform(df)

feature_cols = [f"{c}_ohe" for c in categorical_cols] + [
    f"{c}_imputed" for c in numeric_cols
]
assembler = VectorAssembler(
    inputCols=feature_cols, outputCol="features", handleInvalid="keep"
)

## Modèle ML + split + entraînement
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Modle 1 : Logistic Regression (baseline)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
pipeline = Pipeline(stages=indexers + encoders + [imputer, assembler, lr])
model = pipeline.fit(train)
pred = model.transform(test)

auc_eval = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)
acc_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
auc = auc_eval.evaluate(pred)
f1 = f1_eval.evaluate(pred)
acc = acc_eval.evaluate(pred)
print("AUC =", auc)
print("F1 =", f1)
print("ACC =", acc)
pred.select("Churn", "label", "prediction", "probability").show(10, truncate=False)

# Modèle 2 : Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
pipeline_dt = Pipeline(stages=indexers + encoders + [imputer, assembler, dt])
model_dt = pipeline_dt.fit(train)
pred_dt = model_dt.transform(test)

auc_dt = auc_eval.evaluate(pred_dt)
f1_dt = f1_eval.evaluate(pred_dt)
acc_dt = acc_eval.evaluate(pred_dt)
print("AUC (Decision Tree) =", auc_dt)
print("F1 (Decision Tree) =", f1_dt)
print("ACC (Decision Tree) =", acc_dt)
pred_dt.select("Churn", "label", "prediction", "probability").show(10, truncate=False)

# Modèle 3 : Random Forest Classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label")
pipeline_rf = Pipeline(stages=indexers + encoders + [imputer, assembler, rf])
model_rf = pipeline_rf.fit(train)
pred_rf = model_rf.transform(test)

auc_rf = auc_eval.evaluate(pred_rf)
f1_rf = f1_eval.evaluate(pred_rf)
acc_rf = acc_eval.evaluate(pred_rf)
print("AUC (Random Forest) =", auc_rf)
print("F1 (Random Forest) =", f1_rf)
print("ACC (Random Forest) =", acc_rf)
pred_rf.select("Churn", "label", "prediction", "probability").show(10, truncate=False)

## Sauvegarde des résultats
### LogisticRegression
out_pred = "hdfs://localhost:9000/churn/output/predictions_lr"
pred.select("label", "prediction", "probability").write.mode("overwrite").parquet(
    out_pred
)
# Sauver le model
out_model = "hdfs://localhost:9000/churn/output/model_lr"
model.write().overwrite().save(out_model)
print("Saved predictions to:", out_pred)
print("Saved model to:", out_model)


# Save Decision Tree predictions
out_pred_dt = "hdfs://localhost:9000/churn/output/predictions_dt"
pred_dt.select("label", "prediction", "probability").write.mode("overwrite").parquet(
    out_pred_dt
)

# Save Decision Tree model
out_model_dt = "hdfs://localhost:9000/churn/output/model_dt"
model_dt.write().overwrite().save(out_model_dt)

print("Saved DT predictions to:", out_pred_dt)
print("Saved DT model to:", out_model_dt)

# Save Random Forest predictions
out_pred_rf = "hdfs://localhost:9000/churn/output/predictions_rf"
pred_rf.select("label", "prediction", "probability").write.mode("overwrite").parquet(
    out_pred_rf
)

# Save Random Forest model
out_model_rf = "hdfs://localhost:9000/churn/output/model_rf"
model_rf.write().overwrite().save(out_model_rf)

print("Saved RF predictions to:", out_pred_rf)
print("Saved RF model to:", out_model_rf)
spark.stop()
