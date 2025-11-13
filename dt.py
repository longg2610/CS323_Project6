import random
d = 3
classes = [-1, 0, 1]
features = ["", "Pregnancies" , "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction" , "Age"]     # input follows this format
x = (-1, 5, 88, 66, 150, 24.4, 0.342, 30)       

G = [-1,1,0,1,0,1,0,1,0]  # leaf  1 has class 1, leaf 2 class 0,...
H = [-1, 6, 2, 4, 3, 1, 5, 7]  #H(1) = variance, etc.
w = [-1 , 0.15, 120, 20, 70, 5, 28, 30]


"""
get the i'th least significant bit of number
"""
def get_bit(number, i):
    return ((number >> (i-1)) & 0x1)


def get_random_shares(combination):
    A_share = random.sample(range(2), 1)[0]
    B_share = A_share ^ combination
    return (A_share, B_share)

def get_z_H_values(): 
    x_H = [-1]
    for attribute_index in H[1:]:
        x_H.append(x[attribute_index])

    print(w)
    print(x_H)

    z_i = [-1]
    for i in range(1, len(H)):
        z_i.append(int(x_H[i] >= w[i]))
    print(z_i)

    z_i_shares = [get_random_shares(z) for z in z_i]
    print(z_i_shares)
    return z_i_shares
    
z_i_shares = get_z_H_values()           # z_i is good
print("z_i: " + str(z_i_shares))

def multiply_tuple(tuple1, tuple2):
    return tuple(x * y for x, y in zip(tuple1, tuple2))

for j in range(2**d - 1):
    print(j)
    b = G[j + 1]
    y_j_r = (0, b)
    u = 1
    s = d
    while(s > 0):
        print("y_j_r: " + str(y_j_r))
        print("z" + str(u) + ": " + str(z_i_shares[u]))
        print("j_s: ", str((0, get_bit(j, s))))
        y_j_r = multiply_tuple(y_j_r, (z_i_shares[u] + (0, get_bit(j, s))))
        u = 2*u + get_bit(j, s)
        s -= 1
    print(y_j_r)
    

        
