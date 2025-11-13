import random
d = 3
classes = ["", "Yes", "No"]
    
H = [-1, 6, 2, 4, 3, 1, 5, 7]  #H(1) = variance, etc.
w = [-1 , 0.15, 120, 20, 70, 5, 28, 30]
G = [-1,1,0,1,0,1,0,1,0]  # leaf  1 has class 1, leaf 2 class 0,...

features = ["", "Pregnancies" , "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction" , "Age"]     # input follows this format
x = (-1, 4, 119, 69, 19, 30, 0.1, 29)   


# Class Example instance
# d = 2
# G = [-1,0,1,0,1]
# H = [-1,2,1,3]
# w = [-1,1,1,1]
# x = (-1,0,0,0)


def parse(filename):
    

"""
get the i'th least significant bit of number, i starts from 1 
"""
def get_bit(number, i):
    return ((number >> (i-1)) & 0x1)

"""
randomly generate share of one party and get share of the other by XOR-ing the random value with the output
"""
def get_random_shares(combination):
    A_share = random.sample(range(2), 1)[0]
    B_share = A_share ^ combination
    return [A_share, B_share]

def get_z_H_values(): 
    x_H = [-1]
    for attribute_index in H[1:]:
        x_H.append(x[attribute_index])
    # print(w)
    # print(x_H)
    z_i = [-1]
    for i in range(1, len(H)):
        z_i.append(int(x_H[i] >= w[i]))
    # print(z_i)
    z_i_shares = [get_random_shares(z) for z in z_i]
    # print(z_i_shares)
    return z_i_shares

def main():
    z_i_shares = get_z_H_values()           # z_i is good
    # print("z_i: ", z_i_shares)

    sigma_sum = []
    for j in range(2**d):
        print(j)
        b = G[j + 1]
        y_j_r = [0, b]
        u = 1
        s = d
        while(s > 0):
            # print(z_i_shares, "shouldnt change")
            # print("y_j_r: ", y_j_r)
            # print("z_", u,  ": ", z_i_shares[u])
            # print("j_",s," ", get_bit(j, s))
            z_sum = z_i_shares[u].copy()
            z_sum[1]  = (z_sum[1] + get_bit(j,s)) % 2   # add j_s to Bobby's bit
            # print("SUM ", z_sum)
            y_j_r = (y_j_r[0] ^ y_j_r[1]) * (z_sum[0] ^ z_sum[1])     # multiply XOR of shares of y_j_r with XOR of shares of z_u + j_s  
            # print("y_j_r end value: ", y_j_r)
            y_j_r = get_random_shares(y_j_r)

            # update u and s
            u = 2*u + get_bit(j, s)
            s -= 1

        # print("End Loop: Shares of y_j_r:", y_j_r)
        sigma_sum.append(y_j_r)
    print(sigma_sum)
    class_label = [x[0]^x[1] for x in sigma_sum]
    print ("Class Label", classes[sum(class_label) + 1])
        

main()     
