"""PySpark feature engineering — the defensible "3M+ records with PySpark/SQL" story.

Runs in Spark LOCAL mode (`local[*]`) so it works on a laptop: Spark partitions and
spills to disk, handling the Home Credit auxiliary tables (or Freddie Mac's monthly
performance rows) far beyond what fits in RAM.

Demonstrates, on purpose, the things an interviewer probes:
  * groupBy / agg aggregations across one-to-many tables
  * Window functions (partitionBy + orderBy + lag / rolling counts)
  * Spark SQL via temp views (to show SQL, not just the DataFrame API)
  * a printed row count proving 3M+ rows were processed
  * Parquet output to data/interim/ (the handoff to the Pandas modeling code)

Requires Java 17+ with JAVA_HOME set and `pip install pyspark`.

    python -m src.features.spark_features
    python -m src.features.spark_features --source freddie --path data/raw/freddie_mac
"""
from __future__ import annotations

import argparse

from src.config import CFG


def build_spark(app_name: str | None = None):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name or CFG.spark_app_name)
        .master("local[*]")
        .config("spark.driver.memory", CFG.spark_driver_memory)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def home_credit_features(spark, raw_dir=None):
    """Aggregate bureau + previous_application into per-applicant features.

    Returns (features_df, rows_processed). `rows_processed` counts the raw
    one-to-many rows scanned, which is the honest basis for the 3M+ claim.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    raw = (raw_dir or (CFG.raw / "home_credit")).__str__()
    app = spark.read.csv(f"{raw}/application_train.csv", header=True, inferSchema=True)
    bureau = spark.read.csv(f"{raw}/bureau.csv", header=True, inferSchema=True)
    prev = spark.read.csv(f"{raw}/previous_application.csv", header=True, inferSchema=True)

    rows_processed = app.count() + bureau.count() + prev.count()

    # --- bureau aggregates (groupBy / agg) ---
    bureau_agg = bureau.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("BUREAU_CNT"),
        F.sum(F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)).alias("BUREAU_ACTIVE_CNT"),
        F.avg("AMT_CREDIT_SUM_DEBT").alias("BUREAU_DEBT_MEAN"),
        F.sum("AMT_CREDIT_SUM_OVERDUE").alias("BUREAU_OVERDUE_SUM"),
        F.max("CREDIT_DAY_OVERDUE").alias("BUREAU_MAX_DAYS_OVERDUE"),
    )

    # --- window function: recency rank of each prior application per applicant ---
    w = Window.partitionBy("SK_ID_CURR").orderBy(F.col("DAYS_DECISION").desc())
    prev_ranked = prev.withColumn("recency_rank", F.row_number().over(w))
    prev_agg = prev_ranked.groupBy("SK_ID_CURR").agg(
        F.count("*").alias("PREV_CNT"),
        F.avg(F.when(F.col("NAME_CONTRACT_STATUS") == "Refused", 1.0).otherwise(0.0)).alias(
            "PREV_REFUSED_RATE"
        ),
        F.first(F.when(F.col("recency_rank") == 1, F.col("AMT_CREDIT"))).alias(
            "PREV_LAST_AMT_CREDIT"
        ),
    )

    # --- Spark SQL on a temp view (shows SQL skill explicitly) ---
    app.createOrReplaceTempView("application")
    app_sql = spark.sql(
        """
        SELECT *,
               AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL / 12.0, 0)        AS DTI,
               AMT_CREDIT  / NULLIF(AMT_INCOME_TOTAL, 0)               AS CREDIT_INCOME_RATIO,
               CAST(-DAYS_BIRTH AS DOUBLE) / 365.25                    AS AGE_YEARS
        FROM application
        """
    )

    features = (
        app_sql.join(bureau_agg, on="SK_ID_CURR", how="left")
        .join(prev_agg, on="SK_ID_CURR", how="left")
        .fillna(
            {
                "BUREAU_CNT": 0,
                "BUREAU_ACTIVE_CNT": 0,
                "BUREAU_OVERDUE_SUM": 0.0,
                "BUREAU_MAX_DAYS_OVERDUE": 0,
                "PREV_CNT": 0,
                "PREV_REFUSED_RATE": 0.0,
            }
        )
    )
    return features, rows_processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["home_credit", "freddie"], default="home_credit")
    ap.add_argument("--path", default=None)
    args = ap.parse_args()

    spark = build_spark()
    try:
        if args.source == "freddie":
            raise NotImplementedError(
                "Point this at your downloaded Freddie Mac quarters; the Home Credit path is the "
                "default runnable demo. The same groupBy/window/SQL pattern applies to the monthly "
                "performance files (partitionBy loan id, orderBy reporting month, lag for roll rates)."
            )
        features, rows = home_credit_features(spark, raw_dir=args.path)
        out = CFG.interim / "credit_features.parquet"
        features.write.mode("overwrite").parquet(str(out))
        print(f"Processed {rows:,} raw rows across 3 tables (the '3M+ records' basis).")
        print(f"Wrote {features.count():,} applicant feature rows -> {out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
