#!/bin/sh

HOST_IP="172.21.1.114"
HOST_PORT="33338"

INTERVAL=1
SRC_IP=""
DEV=""

dhclient eth0
sleep 1
dpkg -l | grep rsync || apt-get update 
apt-get install -y rsync
apt-get install -y netcat-openbsd
sleep 1

while true; do
    ip route get $HOST_IP >/dev/null
    if [ $? -eq 0 ]; then
        SRC_IP=`ip route get $HOST_IP | grep " dev " | sed 's/.* src //'`
        DEV=`ip route get $HOST_IP | grep " dev " | sed 's/.*dev \([^ ][^ ]*\).*/\1/'`

        status=`echo "$SRC_IP" | /bin/nc $HOST_IP $HOST_PORT`
    fi
    sleep $INTERVAL
done
