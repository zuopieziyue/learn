package bayes
import com.huaban.analysis.jieba.JiebaSegmenter
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object BayesTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .registerKryoClasses(Array(classOf[JiebaSegmenter]))
      .set("spark.rcp.message.maxSize", "800")
    val spark = SparkSession
      .builder()
      .appName("Bayes Test")
      .enableHiveSupport()
      .config(conf)
      .getOrCreate()

    //定义结巴分词的方法，传入DF, 输出DF, 多一列seg， 分好的词
    def JiebaSeg(df:DataFrame, colname:String): DataFrame = {
      val segmenter = new JiebaSegmenter()
      val seg = spark.sparkContext.broadcast(segmenter)

      val jieba_udf = udf{(sentence:String)=>
        val segV = seg.value
        segV.process(sentence.toString, SegMode.INDEX)
          .toArray()

      }
    }

  }

}
