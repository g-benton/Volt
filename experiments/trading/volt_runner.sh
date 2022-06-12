#!/bin/bash
cat ../../voltron/data/nasdaq100.txt | while read line 
do
    for test_idx in {0..9}
    do
       python TickerSingleDayGenerator.py --kernel=volt --ntimes=25 --test_idx=${test_idx} --save=True --end_date="2022-01-12" --ticker=${line} --mean=constant 
    done
done
