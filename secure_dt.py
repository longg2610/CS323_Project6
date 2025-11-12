d = 3
classes = [0,1]
features = ["variance", "skewness", "curtosis", "entropy"]

G = [-1,1,0,1,0]  # leaf  1 has class 1, leaf 2 class 0,...
H = ["", "variance", "skewness", "curtosis", "entropy"]  #H(1) = variance, etc.


"""
get the i'th least significant bit of number
"""
def get_bit(number, i):
    return ((number >> i) & 0x1)

for j in range(0, 2**d - 1):
    b = G[j+1] - 1

