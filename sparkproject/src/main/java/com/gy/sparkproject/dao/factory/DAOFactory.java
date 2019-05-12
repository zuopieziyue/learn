package com.gy.sparkproject.dao.factory;

import com.gy.sparkproject.dao.ITaskDAO;
import com.gy.sparkproject.dao.impl.TaskDAOImpl;

/**
 * DAO工厂类
 * @author gongyue
 *
 */
public class DAOFactory {
	
	public static ITaskDAO getTaskDAO() {
		return new TaskDAOImpl();
	}
}
