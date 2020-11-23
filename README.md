# smpi

## Usage:

* Get Madelon data from: http://clopinet.com/isabelle/Projects/NIPS2003/MADELON.zip
* place the data in a `MADELON` folder
* Get and install pymit https://github.com/tud-zih-energy/pymit in your env.
* Get mpi4py.

After cloning the repo run:
```
pip install mpi4py
pip install git+https://github.com/tud-zih-energy/pymit.git
wget http://clopinet.com/isabelle/Projects/NIPS2003/MADELON.zip
unzip MADELON.zip
OMP_NUM_THREADS=1 mpirun -n 8 python hcmi_smpi.py ./MADELON/MADELON/ 20
```

