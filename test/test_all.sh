nosetests *.py
for i in {1..5}
do
    mpirun -n $i nosetests *.py
done
