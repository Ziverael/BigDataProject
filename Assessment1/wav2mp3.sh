#!/bin/bash
###PARAMS###
#1 - directory

# Check if filename is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dir> [ext_from] [ext_to]"
    exit 1
fi
if [ "$2" = "" ];then ext1="mp3";else ext1="$2";fi
if [ "$3" = "" ];then ext2="wav";else ext2="$3";fi

change() {
    #1 - directory
    #2 - ext1
    #3 - ext2
    echo "Replacing for $1"
    for dir in `ls -p $1 | grep '/'`
    do
        subdir="$1$dir"
        change $subdir $2 $3
    done
    for file in `ls $1 | grep ".$2"`
    do
        mv "$1/$file" "$1/${file%.*}.$ext2"
    done
}
change $1 $2 $3



#Get all text files (non-recursive)
# for file in `find $1 -iname "*.mp3"`; do mv $file "${file%.*}.wav";done
