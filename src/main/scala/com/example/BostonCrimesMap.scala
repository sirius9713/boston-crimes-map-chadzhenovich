package com.example

import org.apache.spark.sql.SparkSession

object BostonCrimesMap extends App {
  val spark = SparkSession
    .builder()
    .master( master = "local[*]")
    .getOrCreate()

  val crimeFile = args (0)
  val codesFile = args (1)
  val outputFolder = args (2)

  val crimes = spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv( path = s"$crimeFile")

  val offenseCodes = spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv( path = s"$codesFile")

  import spark.implicits._
  import org.apache.spark.sql.functions.broadcast

  val offenseCodesBroadcast = broadcast(offenseCodes)

  val crimesMain = crimes
    .join(offenseCodesBroadcast, $"CODE" === $"OFFENSE_CODE")

  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.expressions.Window

  val crimesTotalCoor = crimesMain
    .groupBy($"DISTRICT").agg(Map(
    "INCIDENT_NUMBER" -> "count",
    "Lat" -> "avg",
    "Long" -> "avg"
  ))
    .na.fill("00", Seq("DISTRICT"))


  val listAgg = udf((xs: Seq[String]) => xs.mkString(", "))

  val w = Window.partitionBy('DISTRICT).orderBy('count desc)

  val crimesFrequentTypes = crimesMain
    .withColumn("TYPE", split($"NAME", "(?<=^[^ -]*)\\ -")(0))
    .groupBy($"DISTRICT", $"TYPE")
    .count()
    .withColumn("rn", row_number.over(w))
    .filter(col("rn") < 4)
    .groupBy($"DISTRICT")
    .agg(listAgg(collect_list("TYPE")).alias("frequent_crime_types"))
    .na.fill("00", Seq("DISTRICT"))
    .withColumnRenamed("DISTRICT", "DISTRICT_1")

  crimesMain.createOrReplaceTempView("crimesSql")

  val crimesMean = spark
    .sql("select t.DISTRICT, percentile_approx(t.cnt,0.5) as crimes_monthly from (select DISTRICT, count(*) cnt from crimesSql group by DISTRICT, MONTH) t group by t.district order by t.district")
    .na.fill("00", Seq("DISTRICT"))
    .withColumnRenamed("DISTRICT", "DISTRICT_2")

  val crimesFinal = crimesTotalCoor
    .join(crimesFrequentTypes, $"DISTRICT" === $"DISTRICT_1")
    .join(crimesMean, $"DISTRICT" === $"DISTRICT_2")
    .drop($"DISTRICT_1")
    .drop($"DISTRICT_2")
    .withColumnRenamed("count(INCIDENT_NUMBER)", "crimes_total")
    .withColumnRenamed("avg(Lat)", "lat")
    .withColumnRenamed("avg(Long)", "long")
    .orderBy($"DISTRICT")
    .repartition(1)
    .write.parquet(s"$outputFolder")

}
