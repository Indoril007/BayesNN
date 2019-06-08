python bayesnn.py -n jobs4/test_c_$1 -s 10 -e 3000 -i 250 -f 500 -t 10 -T 5 -l 0.0001 -L 400 400 -a elu -S $1 -c 50 -p -1 -1 0 0 0.25 -k -0.1 0.1 -5 -4 -b -0.1 0.1 -5 -4
python bayesnn.py -n jobs4/test_a_$1 -s 10 --reinitialize_weights -e 30000 -i 250 -f 500 -t 500 -T 50 -l 0.0001 -L 400 400 -a elu -S $1 -c 50 -p -1 -1 0 0 0.25 -k -0.1 0.1 -5 -4 -b -0.1 0.1 -5 -4
python bayesnn.py -n jobs4/test_b_$1 -s 10 -e 30000 -i 250 -f 500 -t 500 -T 50 -l 0.0001 -L 400 400 -a elu -S $1 -c 50 -p -1 -1 0 0 0.25 -k -0.1 0.1 -5 -4 -b -0.1 0.1 -5 -4
