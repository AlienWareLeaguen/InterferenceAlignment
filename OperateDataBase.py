# -*- coding:utf-8 -*-
# __author__ = 'CaoRui'
import MySQLdb

class HeroDb:
    def __init__(self,name,conn,cur):
        self.name=name
        self.conn=conn
        self.cur=cur
        try:
            # cur.execute('create database if not exists '+name)
            conn.select_db(name)
            conn.commit()
        except MySQLdb.Error,e:
            print "Mysql Error %d: %s" % (e.args[0], e.args[1])

     #create table
    def createTable(self,name):
        try:
            ex=self.cur.execute
            if ex('show tables')==0:
                ex('create table '+name+'(id int,name varchar(20),sex int,age int,info varchar(50))')
                self.conn.commit()
        except MySQLdb.Error,e:
            print "Mysql Error %d : %s" % (e.args[0],e.args[1])

     #insert single record
    def insert(self,name,value):
        try:
            self.cur.execute('insert into '+name+' values(%s,%s,%s,%s,%s)',value)
        except MySQLdb.Error, e:
            print "Mysql Error %d : %s" % (e.args[0], e.args[1])

     #insert more records
    def insertMore(self,name,values):
        try:
            self.cur.executemany('insert into '+name+' values(%s,%s,%s,%s,%s)',values)
        except MySQLdb.Error, e:
            print "Mysql Error %d : %s" % (e.args[0], e.args[1])

     #update sungle record
    def updateSingle(self, name, value):
        try:
            self.cur.execute('update ' + name + ' set name=%s, sex=%s, age=%s, info=%s where id=%s;', value)
        except MySQLdb.Error, e:
            print "Mysql Error %d: %s" % (e.args[0], e.args[1])

     #select all records
    def selectAll(self,name):
        try:
            self.cur.execute('select * from '+name+';')
            results=self.cur.fetchall()
            return  results
        except MySQLdb.Error,e:
            print "Mysql Error %d: %s" % (e.args[0],e.args[1])

    def __del__(self):
        if self.cur!=None:
            self.cur.close()
        if self.conn!=None:
            self.conn.close()
