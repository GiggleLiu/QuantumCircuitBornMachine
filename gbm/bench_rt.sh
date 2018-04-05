nohup mpirun -n 3 python -u program.py chowliu &
for i in {1..10}
    do
        nohup mpirun -n 3 python -u program.py rt $i &
    done

