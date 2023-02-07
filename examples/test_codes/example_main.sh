#!/usr/bin/bash

#PARAMETERS


#coupling parameters
t=1 #1
U=1 #2
V=1 #3


for t in {1,1.5,2};
do
python3 example1.py $t $U $V
python3 example2.py $t $U $V
done

t=1 #1
U=1 #2
V=1 #3

for U in {-1,-0.5,0,0.5,1,1.5};
do
python3 example1.py $t $U $V
python3 example2.py $t $U $V
done