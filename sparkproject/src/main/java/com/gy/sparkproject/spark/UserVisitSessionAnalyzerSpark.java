package com.gy.sparkproject.spark;

import javax.security.auth.login.Configuration;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.hive.HiveContext;

import scala.Tuple2;

import com.alibaba.fastjson.JSONObject;
import com.gy.sparkproject.conf.ConfigurationManager;
import com.gy.sparkproject.constant.Constants;
import com.gy.sparkproject.dao.ITaskDAO;
import com.gy.sparkproject.dao.factory.DAOFactory;
import com.gy.sparkproject.domain.Task;
import com.gy.sparkproject.test.MockData;
import com.gy.sparkproject.util.ParamUtils;

/**
 * 用户访问Session分析spark作业
 * @author gongyue
 *
 */
@SuppressWarnings("unused")
public class UserVisitSessionAnalyzerSpark {
	
	public static void main(String[] args) {
		// 构建spark上下文
		SparkConf conf = new SparkConf()
					.setAppName(Constants.SPARK_APP_NAME_SESSION)
					.setMaster("local");
		
		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContext = getSQLContext(sc.sc());
		
		// 生成模拟测试数据
		mockData(sc, sqlContext);
		
		// 创建需要使用的DAO组件
		ITaskDAO taskDAO = DAOFactory.getTaskDAO();
		
		// 首先得查询出来指定的任务,并获取这个任务的查询参数
		long taskid = ParamUtils.getTaskIdFromArgs(args, null);
		Task task = taskDAO.findById(taskid);
		JSONObject taskParam = JSONObject.parseObject(task.getTaskParam());
		
		// 如果进行session粒度的数据聚合，首先要从user_visit_action表中，查询出来指定日期范围内的行为数据
		JavaRDD<Row> actionDF = getActionRDDByDateRange(sqlContext, taskParam);
		
		//首先，将行为数据，按照session_id进行groupByKey分组
		
		
		
		
		
		
		
		
		// 关闭Spark上下文
		sc.close();
	}
	
	/**
	 * 获取SQLContext
	 * @param sc
	 * @return
	 */
	private static SQLContext getSQLContext(SparkContext sc) {
		boolean local = ConfigurationManager.getBoolean(Constants.SPARK_LOCAL);
		if (local) {
			return new SQLContext(sc);
		}
		else {
			return new HiveContext(sc);
		}
	}
	
	/**
	 * 生成模拟数据 （只有本地模式，才会生成模拟数据）
	 * @param sc
	 * @param sqlContext
	 */
	private static void mockData(JavaSparkContext sc, SQLContext sqlContext) {
		boolean local = ConfigurationManager.getBoolean(Constants.SPARK_LOCAL);
		if (local) {
			MockData.mock(sc, sqlContext);
		}
	}
	
	/**
	 * 获取指定时间范围内的用户访问行为数据
	 * @param sqlContext SQLContext
	 * @param taskParam 任务参数
	 * @return 行为数据RDD
	 */
	private static JavaRDD<Row> getActionRDDByDateRange(
			SQLContext sqlContext, JSONObject taskParam) {
		String startDate = ParamUtils.getParam(taskParam, Constants.PARAM_START_DATE);
		String endDate = ParamUtils.getParam(taskParam, Constants.PARAM_END_DATE);
		
		String sql = 
				"select * "
				+ "from user_visit_action "
				+ "where date>='" + startDate + "' "
				+ "and date<='" + endDate + "'";
		DataFrame actionDF = sqlContext.sql(sql);
		return actionDF.javaRDD();
	}

	/**
	 * 对行为数据按照session粒度进行聚合
	 * @param actionRDD 行为数据
	 * @return session粒度聚合数据
	 */
	private static JavaPairRDD<String, String> aggregateBySession(
			JavaRDD<Row> actionRDD) {
		
		JavaPairRDD<String, Row> sessionid2ActionRDD = actionRDD.mapToPair(
				new PairFunction<Row, String, Row>() {
					
					private static final long serialVersionUID = 1L;
					
					@Override
					public Tuple2<String, Row> call(Row row) throws Exception {
						return new Tuple2<String, Row>(row.getString(2), row);
					}
				});
		
		// 对行为数据按session粒度进行分组
		JavaPairRDD<String, Iterable<Row>> sessionid2ActionsRDD = sessionid2ActionRDD.groupByKey();
		
		// 对每个session分组进行聚合， 将session中所有的搜索词和点击品类都聚合起来
		
		
		
		return null;;
	}
}
