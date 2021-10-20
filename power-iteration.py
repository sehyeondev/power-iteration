import numpy as np

M = np.array([[1,1,1], [1,2,3], [1,3,6]])
# print(M.__repr__())
# # print(M.shape)
x = np.ones((3))
# # x = np.array([1,2,3])
# print (x.__repr__())

# M = np.array([[3,2], [2, 6]])
# x = np.ones(2)

# print (x.shape)
for i in range(10):
    x = np.matmul(M, x) # compute new x
    print (x.__repr__())
    print (abs(x).__repr__())
    print(max(abs(x)))
    
    # x = x/max(abs(x))
    x = x/np.linalg.norm(x)
    # print(x.shape)
    print (x.__repr__())
    print()

new_lambda = np.matmul(np.matmul(np.transpose(x), M), x)
print (new_lambda)