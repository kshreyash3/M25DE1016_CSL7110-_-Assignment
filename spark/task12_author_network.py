from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, to_date, year
import os

# Start Spark session
spark = SparkSession.builder \
    .appName("Task12_Author_Network") \
    .getOrCreate()

# Silence Spark logs for clean output
spark.sparkContext.setLogLevel("ERROR")


# 1. Preprocessing


# Load the dataset
rdd = spark.sparkContext.wholeTextFiles("books/*.txt")
rdd_formatted = rdd.map(lambda x: (os.path.basename(x[0]), x[1]))
books_df = spark.createDataFrame(rdd_formatted, ["file_name", "text"])

# Extract author and release date as strings first
df_extracted = books_df \
    .withColumn("author", regexp_extract(col("text"), r"Author:\s*(.+)", 1)) \
    .withColumn("release_date", regexp_extract(col("text"), r"Release Date:\s*(.+)", 1))

# Extract the 4-digit year as a string
df_extracted = df_extracted.withColumn("release_year", regexp_extract(col("release_date"), r"(\d{4})", 1))

# CLEAN FIRST: Filter out rows where the author or the year is an empty string
df_clean = df_extracted.filter((col("author") != "") & (col("release_year") != ""))

# CAST SECOND: Now that we know there are no empty strings, we can safely cast to integer
df_clean = df_clean.withColumn("release_year", col("release_year").cast("int")) \
                   .select("author", "release_year") \
                   .distinct() # Avoid duplicate identical book entries for the same author/yeartical book entries for the same author/year

# 2. Influence Network Construction

X = 5 # Time window in years

# Alias the dataframe to perform a self-join
df1 = df_clean.alias("author1")
df2 = df_clean.alias("author2")

# Define the influence relationship: 
# author1 influenced author2 if author2 published after author1, but within X years.
# We also ensure an author doesn't influence themselves.
influence_edges = df1.join(df2, 
    (col("author1.author") != col("author2.author")) & 
    (col("author2.release_year") > col("author1.release_year")) & 
    (col("author2.release_year") <= col("author1.release_year") + X)
).select(
    col("author1.author").alias("influencer"), 
    col("author2.author").alias("influenced")
).distinct() # Keep only unique edges between two authors


# 3. Analysis (Degrees)


print("\n" + "="*50)
print(" TASK 12: AUTHOR INFLUENCE NETWORK")
print("="*50 + "\n")

# Calculate Out-Degree (Authors who influenced the most people)
print("TOP 5 AUTHORS WITH HIGHEST OUT-DEGREE (Influencers):")
print("-" * 55)
out_degree = influence_edges.groupBy("influencer").count().withColumnRenamed("count", "out_degree")
top_out = out_degree.orderBy(col("out_degree").desc()).limit(5)
for row in top_out.collect():
    print(f"  {row['influencer']} (Influenced {row['out_degree']} authors)")
print("\n")

# Calculate In-Degree (Authors who were influenced by the most people)
print("TOP 5 AUTHORS WITH HIGHEST IN-DEGREE (Influenced):")
print("-" * 55)
in_degree = influence_edges.groupBy("influenced").count().withColumnRenamed("count", "in_degree")
top_in = in_degree.orderBy(col("in_degree").desc()).limit(5)
for row in top_in.collect():
    print(f"  {row['influenced']} (Influenced by {row['in_degree']} authors)")
print("\n" + "="*50 + "\n")

spark.stop()