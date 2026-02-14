from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, length, avg
import os

# Start Spark session
spark = SparkSession.builder \
    .appName("Task10_Metadata_Analysis") \
    .getOrCreate()

sc = spark.sparkContext

# 1. SILENCE SPARK LOGS (This removes the wall of INFO text)
sc.setLogLevel("ERROR")

# Read whole text files
rdd = sc.wholeTextFiles("books/*.txt")

# Format RDD and create DataFrame
rdd_formatted = rdd.map(lambda x: (os.path.basename(x[0]), x[1]))
books_df = spark.createDataFrame(rdd_formatted, ["file_name", "text"])

# Metadata Extraction
df_extracted = books_df \
    .withColumn("title", regexp_extract(col("text"), r"Title:\s*(.+)", 1)) \
    .withColumn("release_date", regexp_extract(col("text"), r"Release Date:\s*(.+)", 1)) \
    .withColumn("language", regexp_extract(col("text"), r"Language:\s*(.+)", 1)) \
    .withColumn("encoding", regexp_extract(col("text"), r"Character set encoding:\s*(.+)", 1))

# Extract year
df_extracted = df_extracted.withColumn("release_year", regexp_extract(col("release_date"), r"(\d{4})", 1))


print("\n" + "="*50)
print("TASK 10: METADATA EXTRACTION & ANALYSIS")
print("="*50 + "\n")

# --- Books Released Each Year ---
print("BOOKS RELEASED PER YEAR:")
print("-" * 30)
books_per_year = df_extracted.filter(col("release_year") != "").groupBy("release_year").count().orderBy("release_year")
# Collect and print dynamically instead of using the default Spark table
for row in books_per_year.collect():
    print(f"   ➤ Year {row['release_year']}: {row['count']} book(s)")
print("\n")

# --- Most Common Language ---
print("MOST COMMON LANGUAGE:")
print("-" * 30)
most_common_language = df_extracted.filter(col("language") != "").groupBy("language").count().orderBy(col("count").desc()).first()
if most_common_language:
    print(f"   ➤ {most_common_language['language']} (Total: {most_common_language['count']} book(s))\n")

# --- Average Length of Book Titles ---
print("AVERAGE BOOK TITLE LENGTH:")
print("-" * 30)
df_with_title_length = df_extracted.filter(col("title") != "").withColumn("title_length", length(col("title")))
avg_length = df_with_title_length.select(avg("title_length")).first()
if avg_length and avg_length[0] is not None:
    print(f"   ➤ {avg_length[0]:.2f} characters\n")

print("="*50 + "\n")

spark.stop()