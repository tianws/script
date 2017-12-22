#!/bin/bash

ip=172.21.1.114
port=33338
while [ "$ret" != "1" ];  
do  
ret=`ping $ip -c 1 | grep "ttl=" | wc -l`
done  
echo "network ready!" | /bin/nc $ip $port
/home/nvidia/start_project.sh
