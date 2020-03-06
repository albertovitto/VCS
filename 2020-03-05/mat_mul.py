import numpy as np

a = np.random.rand(3, 2)
b = np.random.rand(2, 3)

if a.shape[1] != b.shape[0]:
    print("can't calculate mat mul")
    pass
else:
    print("can calculate mat mul")
    out = np.ndarray(shape=(a.shape[0], b.shape[1]), dtype=float)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for l in range(a.shape[1]):
                out[i][j] = out[i][j] + a[i][l] * b[l][j]
    print(out)
    print(np.dot(a, b))


# Here is some feedback on your code:

#     Using inputs:

#     a = [[0.38906602 0.55236646 0.83786107 0.27444924]
#      [0.67092099 0.48891153 0.22764754 0.87948221]
#      [0.75310489 0.33554379 0.59176635 0.91043679]]

#     b = [[0.09366272 0.0679566 ]
#      [0.92116658 0.76883474]
#      [0.37107343 0.79591726]
#      [0.087392   0.45724224]]

#     'out' is inaccurate
#     Execution time: 0.1 s -- Time limit: 10.0 s

# Your code printed the following output:

# can calculate mat mul
# [[ 8.80155157e-01  1.24347600e+00]
#  [ 6.74542911e-01 -7.26884566e+27]
#  [ 6.78783243e-01  1.19644337e+00]]


#     a = [[0.00214537 0.08301554 0.71122368 0.83631123]]

#     b = [[0.44555345]
#      [0.44460278]
#      [0.72195247]
#      [0.47502278]]

#     'out' is inaccurate
#     Execution time: 0.1 s -- Time limit: 10.0 s

# Your code printed the following output:

# can calculate mat mul
# [[0.9564139]]
