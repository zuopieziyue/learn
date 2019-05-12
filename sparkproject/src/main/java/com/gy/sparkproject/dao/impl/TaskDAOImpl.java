package com.gy.sparkproject.dao.impl;

import com.gy.sparkproject.dao.ITaskDAO;
import com.gy.sparkproject.domain.Task;

/**
 * DAO实现类
 * @author gongyue
 *
 */
public class TaskDAOImpl implements ITaskDAO {

	/**
	 * 根据主键查询任务
	 * @param taskid主键
	 * @return 任务
	 */
	public Task findById(long taskid) {
		final Task task = new Task();
		return task;
	}
	
}
