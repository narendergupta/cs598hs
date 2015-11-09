#!/bin/bash

for (( i=1; i<=20; i++ ))
do
    python3 ../src/get_url_for_university.py --university_list_file ../data/uni_list.csv
    sleep 1
done
