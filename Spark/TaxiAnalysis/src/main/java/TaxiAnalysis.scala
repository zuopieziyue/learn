import jdk.nashorn.internal.runtime.regexp.joni.constants.StringType
import org.apche.spark._
import org.apche.spark.sql._
import org.apche.spark.types._
import org.apche.spark.runctions._
import org.apche.spark.ml.feature.VectorAssembler
import org.apche.spark.ml.clustering.KMeans

val fieldSchema = StructType(Array(
  StructField("TID", StringType, true),
  StructField("Lat", DoubleType, true),
  StructField("Lon", DoubleType, true),
  StructField("Time", StringType, true)
))





