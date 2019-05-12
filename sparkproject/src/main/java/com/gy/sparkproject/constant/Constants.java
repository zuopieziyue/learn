package com.gy.sparkproject.constant;

/**
 * 常量相关接口
 * @author gongyue
 *
 */
public interface Constants {

	/**
	 * 项目配置相关的常量
	 */
	String JDBC_DRIVER="jdbc.driver"; 
	
	String SPARK_LOCAL = "spark.local";
	
	
	/**
	 * Spark作业相关的常量
	 */
	String SPARK_APP_NAME_SESSION = "UserVisitSessionAnalyzerSpark";
	
	/**
	 * 任务相关的常量
	 */
	String PARAM_START_DATE = "startDate";
	String PARAM_END_DATE = "endDate";
	
}
