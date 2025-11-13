import random
d = 3
classes = [-1, 0, 1]
features = ["", "Pregnancies" , "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction" , "Age"]     # input follows this format
x = (-1, 5, 88, 66, 23, 24.4, 0.342, 30)       

G = [-1,1,0,1,0,0,1,0,1]  # leaf  1 has class 1, leaf 2 class 0,...
H = [-1, 6, 2, 4, 3, 1, 4, 7]  #H(1) = variance, etc.
w = [-1 , 0.15, 120, 70, 20, 5, 28, 30]


"""
get the i'th least significant bit of number
"""
def get_bit(number, i):
    return ((number >> i) & 0x1)


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
    
get_z_H_values()



