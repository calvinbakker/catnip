
## Conventions

This library follows specific conventions for tensor network representations and indexing.

### Site indexing
Sites in the system are counted from left to right:

```
0  1  2  3  4  ...  n
o--o--o--o--o--   --o 
|  |  |  |  |       |
```

### State representation
Each site can be in one of two states, represented as vectors in $\mathbb{R}^2$:

- **0/zero/off**: `[1, 0]`
- **1/one/on**: `[0, 1]`
- **Superposition**: `[a, b]` where `a² + b² = 1`

For multi-site configurations:
- The bit-string `00` corresponds to vector `[1, 0, 0, 0]`
- The bit-string `11` corresponds to vector `[0, 0, 0, 1]`

### Vector notation
Multiple sites combine using Kronecker products:

```
000 = [[1,0], [1,0], [1,0]]
    = [1, 0, 0, 0, 0, 0, 0, 0]  (via 0⊗0⊗0)
```

### Index access
Elements are accessed using binary-to-decimal conversion:

- `000` → index `0`
- `100` → index `1`
- `010` → index `2`
- `110` → index `3`
- `001` → index `4`

Example: `v[1][0][1] = v[5]`, where for this array indexing the Fortran convention is essential.

### Matrix-product ansatz
Operators use this index convention:

```
  #--1    1--#--2   1--#--2    1--#
  |          |         |          |
  0          0         0          0

  1          1         1          1
  |          |         |          |
  #--2    2--#--3   2--#--3    2--#
  |          |         |          |
  0          0         0          0
```

## How-to Guide

### Tensor contraction and re-indexing

#### Step 1: Basic contraction
Given two tensors to contract:

```
tensorA:     tensorB:
  1             1     
  |             |      
  #--2       2--#--3 
  |             |      
  0             0     
```

Contract along shared indices:
```python
td = (0, 1)
tensorC = np.tensordot(tensorA, tensorB, axes=td)
```

#### Step 2: Consolidating indices and re-indexing
After contraction, indices are renumbered from `tensorA` to `tensorB`:

```
tensorC:
   0
   |
   #--1
   |
3--#--4
   |
   2
```

where the indices left from `tensorA` are listed first, and after that the indices of `tensorB`.

To reshape into target tensor format:
```python
tp = (2, 0, 3, 1, 4)
rs = (...)  # target shape
tensorC.transpose(tp).reshape(rs, order='F')
```
**Important**: Always use `order='F'` in reshape operations for the correct Fortran indexing.

#### Indexing rules
- In example, for the `tp` tuple we put `2` and position `0`. This relation between the position and the index-number is essential for a correct transpose operation.
- For double edges, list tensorA index first, then tensorB index in `tp`.
- Incorrect ordering will result in wrong results.

#### Final result
After proper reshaping:
```
   1    
   |      
2--#--3 
   |      
   0     
```
which gives us a tensor in the right format for matrix-product operators (following our conventions).
