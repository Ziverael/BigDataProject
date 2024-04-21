#!/usr/bin/bash

###PARAMS###
#1 - string embeded with ""

# Check if string is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <fileS>"
    exit 1
fi
#Get dictionary
words=`wget -qO- "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"`

#Clear text
text=`cat $1 | sed 's/[.!?]//g'`

#Loop over word
for wrd in $text 
do 
    if [[ `echo $words | grep -i $wrd` = ""  ]];then echo $wrd;fi
done