#!/usr/bin/bash

anchors_list=`curl -s https://www.techradar.com/best/free-office-software  | grep -o 'href="[^"]*"' | awk -F'"' '{print $2}' | grep http`
for link in $anchors_list
do
    # echo    `curl -sI $link | grep HTTP | cut -d" " -f2`
    if [ "`curl -sI $link | grep HTTP | cut -d' ' -f2`" = "404" ];then echo "Broken link: $link";fi
done