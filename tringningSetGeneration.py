# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import MySQLdb
import numpy as np
from  OperateDataBase import HeroDb

if __name__=="__main__":
    rowNum=100000
    list=np.random.normal(0,1,4)
    try:
        conn=MySQLdb.connect(host='localhost',user='root',passwd='oppo',port=3306,charset='utf8')
        conn.select_db('inteferencealignment')
        cursor=conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS channel")
        cursor.execute("CREATE TABLE IF NOT EXISTS channel(c1 FLOAT,c2 FLOAT,c3 FLOAT,c4 FLOAT,c5 FLOAT,c6 float,c7 FLOAT ,c8 FLOAT ,c9 FLOAT )")
        for i in xrange(0,rowNum):
            list=np.random.randn(1,9)
            cursor.executemany("INSERT INTO channel(c1,c2,c3,c4,c5,c6,c7,c8,c9) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",list.tolist())
            conn.commit()
    except MySQLdb.Error,e:
        print "Mysql Error %d:%s" % (e.args[0],e.args[1])
    finally:
        cursor.close()
        conn.close()




