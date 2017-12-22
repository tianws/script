#!/bin/sh
touch /tmp/time.txt
start=$(date +%s)
echo "nvidia" | sudo -S /home/nvidia/jetson_clocks.sh &&
clocks_end=$(date +%s)
clocks_time=$(( $clocks_end - $start ))
/home/nvidia/project/about_opencv_caffenet_lane_detection/start.sh | /bin/nc 172.21.1.114 3333
project_end=$(date +%s)
project_time=$(( $project_end - $clocks_end ))
echo "jetson_clocks.sh:" $clocks_time "start_project.sh" $project_time >> /tmp/time.txt
