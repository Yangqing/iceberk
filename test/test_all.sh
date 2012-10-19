# direct call
nosetests *.py

# mpi call
for i in {1..5}
do
    mpirun -n $i nosetests *.py
done

# test no mpi case
PYTHONPATH_SAV=$PYTHONPATH
PYTHONPATH=$PWD/nompi:$PYTHONPATH
nosetests *.py
PYTHONPATH=$PYTHONPATH_SAV
