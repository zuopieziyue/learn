package com.gy.sparkproject.dao;

import com.gy.sparkproject.domain.Task;



/**
 * 任务管理DAO接口
 * @author gongyue
 *
 */
public interface ITaskDAO {
	
	/**
	 * 根据主键查询任务
	 *
	 */
	Task findById(long taskid);
	
	
}