# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import MySQLdb
import matplotlib.pyplot as plot
import numpy as np
if __name__=="__main__":
    try:
        conn=MySQLdb.connect(host='localhost',user='root',passwd='oppo',port=3306,charset='utf8')
        conn.select_db('inteferencealignment')
        cursor=conn.cursor()
        cursor.execute("SELECT * FROM channel")
        results=cursor.fetchall()
        # conn.commit()
        num_array=np.array(results)
        plot.hist(num_array,30,normed=True)
        plot.show()
    except MySQLdb.Error,e:
        print "Mysql Error %d:%s" % (e.args[0], e.args[1])
    finally:
        cursor.close()
        conn.close()