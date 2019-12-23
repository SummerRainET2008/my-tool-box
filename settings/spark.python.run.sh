/opt/spark/2.1/bin/spark-submit \
  --master yarn \
  --conf spark.driver.maxResultSize=5G \
  --conf spark.dynamicAllocation.maxExecutors=400 \
  --driver-memory 30g \
  --executor-memory 8g  \
  --py-files /home/txia/include/algorithm.py \
  GenQuerySequence.py

