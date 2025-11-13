'''
Long Pham and Tanvi Shegaonkar
Project 6 : Secure Decision Tree Evaluation
11/14/2025

This program securely evaluates a decision tree with continuous features 
with Secure Multiparty Computation. 

'''
import random
import math
# 

# d = 3
# # 
# # all inputs are indexed from 1. Thus, a placeholder value is inserted at index 0

# # class labels
# classes = ["", "Yes", "No"]
    
# # 
# H = [-1, 6, 2, 4, 3, 1, 5, 7]  #H(1) = variance, etc.
# w = [-1 , 0.15, 120, 20, 70, 5, 28, 30]
# G = [-1,1,0,1,0,1,0,1,0]  # leaf  1 has class 1, leaf 2 class 0,...

# features = ["", "Pregnancies" , "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction" , "Age"]     # input follows this format
# x = (-1, 4, 119, 69, 19, 30, 0.1, 29)   


# Class Example instance
# d = 2
# G = [-1,0,1,0,1]
# H = [-1,2,1,3]
# w = [-1,1,1,1]
# x = (-1,0,0,0)


def parse(filename):
    with open(filename, 'r') as f:
        d = int(f.readline())
        classes = [""] + (f.readline().split())
        features = [""] + (f.readline().split())

        H = [-1] + f.readline().split()
        for i in range (1, len(H)):
            H[i] = features.index(H[i])     # construct mapping
        G = [""] + (f.readline().split())
        for i in range (1, len(G)):
            G[i] = classes.index(G[i]) - 1     # construct mapping

        w = [-1] + list(map(float, f.readline().split()))
        x = [-1] + list(map(float, f.readline().split()))
    f.close()
    return [d, classes, features, H, G, w, x]

inputs = parse("input.txt")
d, classes, features, H, G, w, x = inputs

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

    z_i = [-1]
    for i in range(1, len(H)):
        z_i.append(int(x_H[i] >= w[i]))
    z_i_shares = [get_random_shares(z) for z in z_i]
    return z_i_shares

def main():
    z_i_shares = get_z_H_values()           # z_i is good
    alpha = math.ceil(math.log2(len(classes)-1))
    sigma =  [[[None for _ in range(2)] for _ in range(2**d)] for _ in range(alpha)]
    for j in range(2**d):
        b = G[j + 1]
        for r in range(1, alpha+1):
            y_j_r = [0, get_bit(b, r)]
            sigma[r-1][j] = y_j_r

        u = 1
        s = d
        while(s > 0):
            z_sum = z_i_shares[u].copy()
            z_sum[1]  = (z_sum[1] + get_bit(j,s)) % 2   # add j_s to Bobby's bit -> z_u + j_s
            for r in range(1, alpha + 1):
                y_j_r = sigma[r-1][j]
                sigma[r-1][j] = get_random_shares((y_j_r[0] ^ y_j_r[1]) * (z_sum[0] ^ z_sum[1])) # multiply XOR of shares of y_j_r with XOR of shares of z_u + j_s  

            # update u and s
            u = 2*u + get_bit(j, s)
            s -= 1

    # print(sigma)

    label_bits = []
    for r in range(1, alpha + 1):
        sigma_r = sigma[r-1].copy()
        class_index_bit = sum([x[0]^x[1] for x in sigma_r])
        label_bits.append(class_index_bit)
        # print("bit ", r, ": ", class_index_bit)
    
    class_index = 0
    for i in range(len(label_bits)):
        class_index += (2**i) * label_bits[i]

    print("Instance", x, "is classified as", classes[class_index + 1])

main()     
