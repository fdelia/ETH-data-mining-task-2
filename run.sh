# use:
#   in python:
#   import os
#   alpha = float(os.environ['ALPHA'])
#   beta = float(os.environ['BETA'])
# correct path/filename
# make executable chmod +x run.sh
# ./run.sh

for i in `seq 0.2 0.08 0.9`
do 
    for j in `seq 0.2 0.08 0.9`
    do
        export ALPHA=$i
        export BETA=$j
        echo "${i}  ${j}"
        python2.7 runner.py data/handout_train.txt data/handout_test.txt my_SGD_momentum.py
    done
done
