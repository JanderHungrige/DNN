# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:53:56 2018

@author: 310122653
"""
from send_mail import noticeEMail

urs='ScriptCallback@gmail.com'
psw='$Siegel#1'
fromaddr='ScriptCallback@gmail.com
toaddr='jan.werth@philips.com'
starttime=datetime.now()

# Send notification email
noticeEMail(starttime, usr, psw, fromaddr, toaddr)