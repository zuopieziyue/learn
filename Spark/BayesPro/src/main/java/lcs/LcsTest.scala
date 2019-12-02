package lcs
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object LcsTest {
  def main(args: Array[String]): Unit =
  {
    val spark = SparkSession
      .builder()
      .appName("LCS Test")
      .enableHiveSupport()
      .getOrCreate()

    val df = spark.sql("select a, b from badou.lcs")

    def LCS(a:String, b:String): Double = {
      val n = a.length
      val m = b.length
      val opt = Array.ofDim[Int](n+1, m+1)
      for (i<- (0 until n).reverse) {
        for (j<- (0 until m).reverse) {
          if (a(i) == b(j)) opt(i)(j) = opt(i+1)(j+1) + 1
          else opt(i)(j) = opt(i+1)(j).max(opt(i)(j+1))
        }
      }
      return opt(0)(0)*2/(m+n).toDouble
    }

    val a = "公司的审计结果发生重大变化"
    val b = "公司审计结果发生细小变化"
    println(LCS(a,b))

    val lcsUDF = udf{(a:String, b:String)=>LCS(a, b)}

    df.select("a").join(df.select("b"))
      .withColumn("lcs_sim", lcsUDF(col("a"), col("b")))
      .show()
  }




}
