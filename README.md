# smpi

## Usage:

* Get Madelon data from: http://clopinet.com/isabelle/Projects/NIPS2003/MADELON.zip
* place the data in a `MADELON` folder
* Get and install pymit https://github.com/tud-zih-energy/pymit in your env.
* Get mpi4py.

Run
```
OMP_NUM_THREADS=1 mpirun -n 8 python cmi_test_hcmi_smpi.py ./MADELON/MADELON/ 20
```

