#!/bin/bash
TEXTNAME=$(hostname)"_IP.txt" &&
date > $TEXTNAME &&
hostname -I >> $TEXTNAME &&
smbclient //172.22.52.68/share -U nobody%  <<- EOF
cd IP_Record
put $TEXTNAME
exit
EOF
