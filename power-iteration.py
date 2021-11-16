import numpy as np

def power_iteration(M, x, small_constant):
    prev_x = np.zeros(len(x))
    while np.linalg.norm(x-prev_x) >= small_constant:
        prev_x = x
        x = np.matmul(M, x) # compute new x
        x = x/np.linalg.norm(x)
    return x

# Exercise 11.1.7 in the MMDS textbook
small_constant = 0.000001
M = np.array([[1,1,1], [1,2,3], [1,3,6]])
x = np.ones((3,1)) # vector of three 1's
# (a) find principal eigenvector using power iteration
first_x = power_iteration(M, x, small_constant)
# (b) compute principal eigenvalue
first_lambda = np.dot(np.dot(np.transpose(first_x), M), first_x)
# (c) Construct a new matrix
new_M = M - first_lambda * np.dot(first_x, np.transpose(first_x))
# (d) find second eigenpair for M
second_x = power_iteration(new_M, x, small_constant)
second_lambda = np.dot(np.dot(np.transpose(second_x), M), second_x)
# (e) find the third eigenpair by repeating (c) and (d)
third_M = new_M - second_lambda * np.dot(second_x, np.transpose(second_x))
third_x = power_iteration(third_M, x, small_constant)
third_lambda = np.dot(np.dot(np.transpose(third_x), M), third_x)

print ("(a)")
print ("principal eigenvector")
print (first_x.__repr__())
print ("(b)")
print ("principal eigenvalue")
print (first_lambda)
print ("(c)")
print ("new matrix")
print (new_M.__repr__())
print ("(d)")
print ("second eigenvector")
print (second_x.__repr__())
print ("second eigenvalue")
print (second_lambda)
print ("(e)")
print ("third matrix")
print (third_M.__repr__())
print ("third eigenvector")
print (third_x.__repr__())
print ("third eigenvalue")
print (third_lambda)

# # test if my answer is right
# w, v = np.linalg.eig(M)
# print(w)
# print(v)
