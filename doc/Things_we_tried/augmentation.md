# structure augmentation

### Install Packages

```python
%%capture
!pip install forgi
!yes Y |conda install -c bioconda viennarna
```

### Generating MFE, centroid and MEA structure

```python
import RNA

def compute_structure(seq):
    # create fold_compound data structure (required for all subsequently applied  algorithms)
    fc = RNA.fold_compound(seq)
    # compute MFE and MFE structure
    (mfe_struct, mfe) = fc.mfe()
    # rescale Boltzmann factors for partition function computation
    fc.exp_params_rescale(mfe)
    # compute partition function
    (pp, pf) = fc.pf()
    # compute centroid structure
    (centroid_struct, dist) = fc.centroid()
    # compute MEA structure
    (MEA_struct, MEA) = fc.MEA()
    return mfe_struct,centroid_struct,MEA_struct
```

### augmented_structures

```{figure} Images/augmented_structure.png
---
name: Augmented Structures [MFE, centroid and MEA]
---
Augmented Structures [MFE, centroid and MEA]
```