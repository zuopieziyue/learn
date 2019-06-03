package com.gy.sparkproject.spark;

import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;

import javax.security.auth.login.Configuration;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
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
import com.gy.sparkproject.util.DateUtils;
import com.gy.sparkproject.util.StringUtils;
import com.gy.sparkproject.util.ValidUtils;

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
		JavaRDD<Row> actionRDD = getActionRDDByDateRange(sqlContext, taskParam);
		JavaPairRDD<String, Row> sessionid2actionRDD = getSessionid2ActionRDD(actionRDD);
		
		JavaPairRDD<String, String> sessionid2AggrInfoRDD = 
				aggregateBySession(sqlContext, sessionid2actionRDD);

		// 测试
		System.out.println(sessionid2AggrInfoRDD.count());
		for (Tuple2<String, String> tuple : sessionid2AggrInfoRDD.take(10)) {
			System.out.println(tuple._2);
		}
		
		// 针对session粒度的聚合数据，按照使用者指定的筛选参数进行数据过滤
		JavaPairRDD<String, String> filteredSessionid2AggrInfoRDD = 
				filterSession(sessionid2AggrInfoRDD, taskParam);
		
		// 测试
		System.out.println(filteredSessionid2AggrInfoRDD.count());
		for (Tuple2<String, String> tuple : filteredSessionid2AggrInfoRDD.take(10)) {
			System.out.println(tuple._2);
		}

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
		
		String sql = "select * "
				+ "from user_visit_action "
				+ "where date>='" + startDate +"' "
				+ "and date<='" + endDate +"'";
		
		DataFrame actionDF = sqlContext.sql(sql);
		
		return actionDF.javaRDD();	
	}
	
	/**
	 * 获取sessionid2到访问行为数据的映射的RDD
	 * @param actionRDD 
	 * @return
	 */
	public static JavaPairRDD<String, Row> getSessionid2ActionRDD(JavaRDD<Row> actionRDD) {
		
		return actionRDD.mapPartitionsToPair(new PairFlatMapFunction<Iterator<Row>, String, Row>() {

			private static final long serialVersionUID = 1L;

			@Override
			public Iterable<Tuple2<String, Row>> call(Iterator<Row> iterator)
					throws Exception {
				List<Tuple2<String, Row>> list = new ArrayList<Tuple2<String, Row>>();
				
				while(iterator.hasNext()) {
					Row row = iterator.next();
					list.add(new Tuple2<String, Row>(row.getString(2), row));  
				}
				
				return list;
			}
			
		});
	}
	
	/**
	 * 对行为数据按照session粒度进行聚合
	 * @param acitonRDD 行为数据RDD
	 * @return session粒度聚合数据
	 */
	private static JavaPairRDD<String, String> aggregateBySession(
			SQLContext sqlContext, JavaPairRDD<String, Row> sessionid2ActionRDD) {

		// 对行为数据按session粒度分组
		JavaPairRDD<String, Iterable<Row>> sessionid2ActionsRDD 
			= sessionid2ActionRDD.groupByKey();
		
		// 对每一个session分组进行聚合，将session中多有的搜索词和点击品类都聚合起来
		JavaPairRDD<Long, String> userid2PartAggrInfoRDD = sessionid2ActionsRDD.mapToPair(
				new PairFunction<Tuple2<String, Iterable<Row>>, Long, String>() {
					
					private static final long serialVersionUID = 1L;
					
					@Override
					public Tuple2<Long, String> call(Tuple2<String, Iterable<Row>> tuple) throws Exception {
						String sessionid = tuple._1;
						Iterator<Row> iterator = tuple._2.iterator();
						
						StringBuffer searchKeywordsBuffer = new StringBuffer("");
						StringBuffer clickCategoryIdsBuffer = new StringBuffer("");
						
						Long userid = null;
						
						//session的起始时间和结束时间
						Date startTime = null;
						Date endTime = null;
						//session的访问补偿
						int stepLength = 0;
						
						//遍历session所有的访问行为
						while(iterator.hasNext()) {
							//提取每个访问行为的搜索词字段和点击品类字段
							Row row = iterator.next();
							if (userid == null) {
								userid = row.getLong(1);
							}
							String searchKeyword = row.getString(5);
							Long clickCategoryId = row.getLong(6);
							
							//将搜索词和点击品类id拼接到字符串中去
							if (StringUtils.isNotEmpty(searchKeyword)) {
								if(!searchKeywordsBuffer.toString().contains(searchKeyword)) {
									searchKeywordsBuffer.append(searchKeyword + ",");
								}
							}
							if(clickCategoryId != null) {
								if(!clickCategoryIdsBuffer.toString().contains(String.valueOf(clickCategoryId))) {
									clickCategoryIdsBuffer.append(clickCategoryId + ",");
								}
							}
							
							//计算session开始时间和结束时间
							Date actionTime = com.gy.sparkproject.util.DateUtils.parseTime(row.getString(4));
							
							if(startTime == null) {
								startTime = actionTime;
							}
							if(endTime == null) {
								endTime = actionTime;
							}
							
							if(actionTime.before(startTime)) {
								startTime = actionTime;
							}
							
							if(actionTime.after(endTime)) {
								endTime = actionTime;
							}
							
							//计算session访问步长
							stepLength ++;
						}
						
						String searchKeywords = StringUtils.trimComma(searchKeywordsBuffer.toString());
						String clickCategoryIds = StringUtils.trimComma(clickCategoryIdsBuffer.toString());
						
						String partAggrInfo = Constants.FIELD_SESSION_ID + "=" + sessionid + "|"
								+ Constants.FIELD_SEARCH_KEYWORDS + "=" + searchKeywords + "|" 
								+ Constants.FIELD_CLICK_CATEGORY_IDS + "=" + clickCategoryIds;
						
						return new Tuple2<Long, String>(userid, partAggrInfo);
					}
				});
		
			// 查询所有用户数据，并映射成<userid, Row>的格式
			String sql = "select * from user_info";
			JavaRDD<Row> userInfoRDD = sqlContext.sql(sql).javaRDD();
			
			JavaPairRDD<Long, Row> userid2InfoRDD = userInfoRDD.mapToPair(
					new PairFunction<Row, Long, Row>() {
						
						private static final long serialVersionUID = 1L;
						
						@Override
						public Tuple2<Long, Row> call(Row row) throws Exception {
							return new Tuple2<Long, Row>(row.getLong(0), row);
						}
	
					});
			// 将session粒度聚合数据，与用户信息进行join
			JavaPairRDD<Long, Tuple2<String, Row>> userid2FullInfoRDD = 
					userid2PartAggrInfoRDD.join(userid2InfoRDD);
		
			// 对join起来的数据进行拼接，并且返回<sessionid, fullAggrInfo>格式的数据
			JavaPairRDD<String, String> sessionid2FullAggrInfoRDD = userid2FullInfoRDD.mapToPair(
					new PairFunction<Tuple2<Long, Tuple2<String, Row>>, String, String>() {
					
						private static final long serialVersionUID = 1L;
						
						@Override
						public Tuple2<String, String> call(Tuple2<Long, Tuple2<String, Row>> tuple) throws Exception {
							
							String partAggrInfo = tuple._2._1;
							Row userInfoRow = tuple._2._2;
							
							String sessionid = StringUtils.getFieldFromConcatString(
									partAggrInfo, "\\|", Constants.FIELD_SESSION_ID);
							
							int age = userInfoRow.getInt(3);
							String professional = userInfoRow.getString(4);
							String city = userInfoRow.getString(5);
							String sex = userInfoRow.getString(6);
							
							String fullAggrInfo = partAggrInfo + "|"
									+ Constants.FIELD_AGE + "=" + age + "|"
									+ Constants.FIELD_PROFESSIONAL + "=" + professional + "|"
									+ Constants.FIELD_SEX + "=" + sex;
							
							return new Tuple2<String, String>(sessionid, fullAggrInfo);
						}
				});
		
		return sessionid2FullAggrInfoRDD;
	}
	
	/**
	 * 过滤session数据
	 * @param sessionid2AggrInfoRDD
	 * @return
	 */
	private static JavaPairRDD<String, String> filterSession(
			JavaPairRDD<String, String> sessionid2AggrInfoRDD, 
			final JSONObject taskParam) {
		
		String startAge = ParamUtils.getParam(taskParam, Constants.PARAM_START_AGE);
		String endAge = ParamUtils.getParam(taskParam, Constants.PARAM_END_AGE);
		String processionals = ParamUtils.getParam(taskParam, Constants.PARAM_PROFESSIONALS);
		String citis = ParamUtils.getParam(taskParam, Constants.PARAM_CITIES);
		String sex = ParamUtils.getParam(taskParam, Constants.PARAM_SEX);
		String keywords = ParamUtils.getParam(taskParam, Constants.PARAM_KEYWORDS);
		String categoryIds = ParamUtils.getParam(taskParam, Constants.PARAM_CATEGORYIDS);
		
		String _parameter = (startAge != null ? Constants.PARAM_START_AGE + "=" + startAge + "|" : "")
				+ (endAge != null ? Constants.PARAM_END_AGE + "=" + endAge + "|" : "")
				+ (processionals != null ? Constants.PARAM_PROFESSIONALS + "=" + processionals + "|" : "")
				+ (citis != null ? Constants.PARAM_CITIES + "=" + citis + "|" : "")
				+ (sex != null ? Constants.PARAM_SEX + "=" + sex + "|" : "")
				+ (keywords != null ? Constants.PARAM_KEYWORDS + "=" + keywords + "|" : "")
				+ (categoryIds != null ? Constants.PARAM_CATEGORYIDS + "=" + categoryIds + "|" : "");
		if (_parameter.endsWith("\\|")) {
			_parameter = _parameter.substring(0, _parameter.length() - 1);
		}
	
		final String parameter = _parameter;
		
		// 根据筛选参数进行过滤
		JavaPairRDD<String, String> filteredSessionid2AggrInfoRDD = sessionid2AggrInfoRDD.filter(
				new Function<Tuple2<String, String>, Boolean>() {
					
					private static final long serialVersionUID = 1L;
					
					@Override
					public Boolean call (Tuple2<String, String> tuple) throws Exception {
						//首先，从tuple中获取聚合数据
						String aggrInfo = tuple._2;
						
						//接着，一次按照筛选条件进行过滤
						//按照年龄范围进行过滤（startAge, endAge）
						if(!ValidUtils.between(aggrInfo, Constants.FIELD_AGE, 
								parameter, Constants.PARAM_START_AGE, Constants.PARAM_END_AGE)) {
							return false;
						}
						
						// 按照职业范围进行过滤(processionals)
						if(!ValidUtils.in(aggrInfo, Constants.FIELD_PROFESSIONAL, 
								parameter, Constants.PARAM_PROFESSIONALS)) {
							return false;
						}
						
						// 按照城市范围进行过滤
						if(!ValidUtils.in(aggrInfo, Constants.FIELD_CITY, 
								parameter, Constants.PARAM_CITIES)) {
							return false;
						}
						
						// 按照性别进行过滤
						if(!ValidUtils.equal(aggrInfo, Constants.FIELD_SEX, 
								parameter, Constants.PARAM_SEX)) {
							return false;
						}
						
						// 按照搜索词进行过滤
						if(!ValidUtils.in(aggrInfo, Constants.FIELD_SEARCH_KEYWORDS, 
								parameter, Constants.PARAM_KEYWORDS)) {
							return false;
						}
						
						// 按照点击品类ID进行过滤
						if(!ValidUtils.in(aggrInfo, Constants.FIELD_CLICK_CATEGORY_IDS, 
								parameter, Constants.PARAM_CATEGORYIDS)) {
							return false;
						}
						
						return true;
					}
				});
		
		return filteredSessionid2AggrInfoRDD;
	}

}
