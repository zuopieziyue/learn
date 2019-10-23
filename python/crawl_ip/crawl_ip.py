# -*- coding: utf-8 -*-  
########################  
#author:gongyue  
#date:2017/12/08  
#crawl ip registration location
#use ip138
########################  
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import time, re
from bs4 import BeautifulSoup
import requests

def crawl_ip_registration_location(ip):
    url = 'http://www.ip138.com/ips138.asp?ip=' + ip
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
    result_list = soup.find_all("ul", "ul1")
    print result_list
    f1 = open("ip_result.txt", "w")
    for result in result_list:
        f1.write(result.li.text)
    f1.close()

#主函数
if __name__ == '__main__': 
    print u'查询ip的地址'
    ip = '166.111.139.44'
    crawl_ip_registration_location(ip)
    


