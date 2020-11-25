# smpi


Simple MPI for Python allows MPI parallelisation using python decorators.

```
@smpi.collect(smpi.dist_type.gather)
@smpi.distribute(smpi.dist_type.local, smpi.dist_type.local, smpi.dist_type.scatter)
def calculate_mi(X, Y, features):
    MI = numpy.full([len(features)], numpy.nan, dtype=numpy.float)
    for i,X_i in enumerate(features):
        MI[i] = pymit.I(X[:, X_i], Y , bins=[bins, 2])
    return [MI]
```

The example uses previously distributed numpy Arrays `X` and `Y` and scatters a 1D numpy array `features`.
Than the Mutual Information between the features in `X` and the target variable `Y` is calculated for the given `features`.
The result is gathered and returned.

A more detailed example can be found in `hcmi_smpi.py`.
Details about Mutual Information and Feature Selection can be found at https://github.com/tud-zih-energy/pymit .


## Usage:

For smpi itself, you only need `mpi4py`. To try the example, you need the following:

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

## License

If the LGPL is not good License for you, please feel free to write to andreas.gocht@tu-dresden.de
