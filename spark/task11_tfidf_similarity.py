from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer
from pyspark.sql.types import FloatType
import os

# Start Spark session
spark = SparkSession.builder \
    .appName("Task11_TFIDF_Similarity") \
    .getOrCreate()

# Silence standard info logs for clean output
spark.sparkContext.setLogLevel("ERROR")


# 0. Load Data

rdd = spark.sparkContext.wholeTextFiles("books/*.txt")
rdd_formatted = rdd.map(lambda x: (os.path.basename(x[0]), x[1]))
books_df = spark.createDataFrame(rdd_formatted, ["file_name", "text"])


# 1. Preprocessing

# Remove Project Gutenberg header/footer (using a standard regex pattern for Gutenberg ebooks)
clean_text = regexp_replace(col("text"), r"(?s).*?\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", "")
clean_text = regexp_replace(clean_text, r"(?s)\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*", "")

# Convert to lowercase and remove punctuation (keep only alphabetic characters and spaces)
clean_text = lower(regexp_replace(clean_text, r"[^a-zA-Z\s]", " "))
df_clean = books_df.withColumn("cleaned_text", clean_text)

# Tokenize into words
tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
df_words = tokenizer.transform(df_clean)

# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_words)


# 2. TF-IDF Calculation

# Calculate Term Frequency (TF)
cv = CountVectorizer(inputCol="filtered_words", outputCol="rawFeatures")
cv_model = cv.fit(df_filtered)
df_tf = cv_model.transform(df_filtered)

# Calculate Inverse Document Frequency (IDF) and compute TF-IDF score
idf = IDF(inputCol="rawFeatures", outputCol="tfidf_features")
idf_model = idf.fit(df_tf)
df_tfidf = idf_model.transform(df_tf)


# 3. Book Similarity (Cosine Similarity)

# To calculate Cosine Similarity efficiently, we normalize vectors to length 1.
# The dot product of two L2-normalized vectors gives their cosine similarity.
normalizer = Normalizer(inputCol="tfidf_features", outputCol="norm_features", p=2.0)
df_normalized = normalizer.transform(df_tfidf)

TARGET_BOOK = "2.txt"

# Extract the normalized vector for the target book
target_row = df_normalized.filter(col("file_name") == TARGET_BOOK).select("norm_features").first()

if target_row is None:
    print(f"\nError: '{TARGET_BOOK}' not found in the dataset.\n")
else:
    target_vector = target_row["norm_features"]

    # Create a User Defined Function (UDF) to compute the dot product
    def calculate_cosine_similarity(v1):
        return float(v1.dot(target_vector))

    cosine_sim_udf = udf(calculate_cosine_similarity, FloatType())

    # Apply UDF to calculate similarity for all books
    df_similarity = df_normalized.withColumn("similarity", cosine_sim_udf(col("norm_features")))

    print("\n" + "="*55)
    print(f"TOP 5 MOST SIMILAR BOOKS TO '{TARGET_BOOK}'")
    print("="*55 + "\n")

    # Exclude the target book itself, sort by similarity descending, and get top 5
    top_5_similar = df_similarity.filter(col("file_name") != TARGET_BOOK) \
                                 .select("file_name", "similarity") \
                                 .orderBy(col("similarity").desc()) \
                                 .limit(5)
    
    for row in top_5_similar.collect():
        print(f"   ➤ {row['file_name']} (Cosine Similarity: {row['similarity']:.4f})")
    print("\n")

spark.stop()