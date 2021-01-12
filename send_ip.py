#/bin/env python
# -*-coding:utf8-*-
import socket
import fcntl
import time
import struct
import smtplib
import urllib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import commands

#发送邮件的基本函数，参数依次如下
# smtp服务器地址、邮箱用户名，邮箱第三方密码，发件人地址，收件人地址（列表的方式），邮件主题，邮件html内容
def sendEmail(smtpserver,username,password,sender,receiver,subject,msghtml):
    msgRoot = MIMEMultipart('related')
    msgRoot["To"] = ','.join(receiver)
    msgRoot["From"] = sender
    msgRoot['Subject'] =  subject
    msgText = MIMEText(msghtml,'html','utf-8')
    msgRoot.attach(msgText)
    #sendEmail
    smtp = smtplib.SMTP()
    smtp.connect(smtpserver)
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msgRoot.as_string())
    smtp.quit()
    print subject+" has been sent"

# 检查网络连同性
def check_network():
    while True:
        try:
            result=urllib.urlopen('http://baidu.com').read()
            # print result
            print "Network is Ready!"
            break
        except Exception , e:
           print e
           print "Network is not ready,Sleep 5s...."
           time.sleep(5)
    return True

# 获得本级制定接口的ip地址
def get_ip_address():
    #s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #s.connect(("1.1.1.1",80))
    #print s.getsockname()
    #ipaddr=" ".join(str(d) for d in s.getsockname())
    #s.close()
    hostname = commands.getstatusoutput('hostname')
    hostname = str(hostname[1])
    ipaddr = commands.getstatusoutput('hostname -I')
    #ipaddr=" ".join(str(d) for d in ipaddr)
    ipaddr=str(ipaddr[1])
    content = hostname+" : "+ipaddr
    print content
    return hostname,content

if __name__ == '__main__':
    check_network()
    hostname,content=get_ip_address()
    sendEmail('smtp.163.com','pi_raspberrypi@163.com','111qqq','pi_raspberrypi@163.com',['tianws@mapbar.com','342415221@qq.com'],'IP Address Of '+hostname,content)
