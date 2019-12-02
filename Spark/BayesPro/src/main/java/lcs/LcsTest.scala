package lcs
import org.apache.spark.sql.SparkSession

object LcsTest {
  def main(args: Array[String]): Unit =
  {
    val spark = SparkSession
      .builder()
      .appName("LCS Test")
      .enableHiveSupport()
      .getOrCreate()
  }

  def LCS(a:String, b:String): Double = {
    val n = a.length
    val m = b.length
  }

}
