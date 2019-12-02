package lcs
import org.apache.spark.sql.SparkSession

object LcsTest {
  def main(args: Array[String]): Unit =
  {
//    val spark = SparkSession
//      .builder()
//      .appName("LCS Test")
//      .enableHiveSupport()
//      .getOrCreate()

    def LCS(a:String, b:String): Double = {
      val n = a.length
      val m = b.length
      val opt = Array.ofDim[Int](n+1, m+1)
      for (i<- 0 until n) {
        for (j<- 0 until m) {
          if (a(i) == b(j)) opt(i)(j) = opt(i+1)(j+1) + 1
          else opt(i)(j) = opt(i+1)(j).max(opt(i)(j+1))
        }
      }
      opt(0)(0)
    }

    val a = "ABCBDAB"
    val b = "BDCABA"
    println(LCS(a,b))

  }




}
