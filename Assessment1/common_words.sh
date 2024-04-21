#!/usr/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dir>"
    exit 1
fi

tmp_IFS=$ITS
#Get text
dumped_text=`find $1 -iname "*.txt" -exec cat {} +`
#Normalise data: remove special characters, set uppercase
preprocessed_data=`echo "$dumped_text" | tr -cd '[a-zA-Z] \n' | sed 's/[ \t]\+/\n/g' | grep -v "^$"` 
preprocessed_data=${preprocessed_data^^}
#Create bins for data
bins_counts=`echo "$preprocessed_data" | sort | uniq -c | sort -nr | head -n 10`
echo "Top 10 most frequent words:"
echo "$bins_counts"



