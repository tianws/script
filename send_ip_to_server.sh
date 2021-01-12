#!/bin/bash

timeout=5
target=www.baidu.com
ret_code=0

while [[ "$ret_code"x != "200"x ]]
do
    ret_code=`curl -I -s --connect-timeout $timeout $target -w %{http_code} | tail -n1`
done

TEXTNAME=$(hostname)"_IP.txt" &&
date > $TEXTNAME &&
hostname -I >> $TEXTNAME &&
smbclient //172.22.52.68/share -U nobody%  <<- EOF
cd IP_Record
put $TEXTNAME
exit
EOF
