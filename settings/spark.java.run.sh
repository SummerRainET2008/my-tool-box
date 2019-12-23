/opt/spark/2.1/bin/spark-submit \
  --master yarn \
  --conf spark.driver.maxResultSize=2G \
  --conf spark.dynamicAllocation.maxExecutors=200 \
  --driver-java-options '-Ddata.dir=premium.test' \
  --driver-memory 10g \
  --executor-memory 10g  \
  --class dl.EncoderDecoderLSTM \
  --jars dl4j-examples-0.8-SNAPSHOT.base.jar \
  dl4j-examples-0.8-SNAPSHOT.jar -epochNum 100 -useSparkLocal

