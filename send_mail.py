# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:41:22 2018

@author: 310122653
"""

import smtplib
from datetime import datetime


def noticeEMail(starttime, usr, psw, fromaddr, toaddr,runningNumber,description):
    """
    Sends an email message through GMail once the script is completed.  
    Developed to be used with AWS so that instances can be terminated 
    once a long job is done. Only works for those with GMail accounts.
    
    starttime : a datetime() object for when to start run time clock
    usr : the GMail username, as a string
    psw : the GMail password, as a string 
    
    fromaddr : the email address the message will be from, as a string
    
    toaddr : a email address, or a list of addresses, to send the 
             message to
    """

    # Calculate run time
    runtime=datetime.now() - starttime
    
    # Initialize SMTP server
    server=smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(usr,psw)
    
    # Send email
    senddate=datetime.strftime(datetime.now(), '%Y-%m-%d')
    subject="Your job " + runningNumber + " has completed"
    m="Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer: My-Mail\r\n\r\n" % (senddate, fromaddr, toaddr, subject)
    msg= runningNumber+description+'\n Job runtime: '+str(runtime) + '\n Job start: ' +str(starttime)
    
    
    server.sendmail(fromaddr, toaddr, m+msg)
    server.quit()