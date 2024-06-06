import os
import sys
import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt


def least_squares_cubic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x, (x**2), (x**3)))
    v = equation_thing(x_e, y)
    # print(v)
    return v

#plotting function from lab3 i think
def plot_cubic(xs,ys,n):
    fig, ax = plt.subplots()
    for i in range(n):
        tempxs = xs[i]
        tempys = ys[i]
        a, b, c, d = least_squares_cubic(tempxs, tempys)
        arrangedY = a + b * tempxs + c * (tempxs ** 2) + d*(tempxs **3)
        print(squareError(tempys, arrangedY))
        ax.scatter(tempxs, tempys, s=20)
        ax.plot(tempxs, arrangedY, 'r-', lw=2) 
    plt.show()


#plotting function from lab3 i think
def plot_lin(xs,ys,n):
    fig, ax = plt.subplots()
    for i in range(n):
        tempxs = xs[i]
        tempys = ys[i]
        a, b = least_squares_lin(tempxs, tempys)
        arrangedY = a + b*tempxs
        print(squareError(tempys, arrangedY))
        ax.scatter(tempxs, tempys, s=20)
        ax.plot(tempxs, arrangedY, 'r-', lw=2)
    plt.show()

    # decides which line to use based of squareError ATM
def decider(tempxs, tempys, i):
    l_s_lin = least_squares_lin(tempxs, tempys)
    arrangedY_lin = l_s_lin[0] + l_s_lin[1]*tempxs
    l_s_cub = least_squares_cubic(tempxs, tempys)
    arrangedY_cub = l_s_cub[0] + l_s_cub[1] * tempxs + l_s_cub[2] * (tempxs ** 2) + l_s_cub[3]*(tempxs **3)
    s_e_lin = squareError(tempys, arrangedY_lin)
    s_e_cub = squareError(tempys, arrangedY_cub)
    print("This is the linear squared error: ",s_e_lin)
    print("This is the cubic squared error: ",s_e_cub)
    if (s_e_cub < s_e_lin):
        print("This segment ",(i+1)," is cubic\n")
        return arrangedY_cub
    else:
        print("This segment ",(i+1)," is linear\n")
        return arrangedY_lin



def crossValidation (xs, ys):
    # this bit divides the xs and ys array into two sepperate groups for testing 
    indexes = []
    ammountOfSamplesInTest = 10
    notInIndexes = True
    for i in range(ammountOfSamplesInTest):
        j = random.randint(0,19)
        for h in range(len(indexes)):
            if indexes[h] == j:
                ammountOfSamplesInTest = ammountOfSamplesInTest + 1
                notInIndexes = False
        if notInIndexes:
            indexes.append(j)
        notInIndexes = True
    print(indexes)
    #these are the two testing arrays of length 5   
    testxs = np.array(xs[indexes])
    testys = np.array(ys[indexes])
    # taking away the elements in the test set from the whole set
    trainxs = np.delete(xs, indexes)
    trainys = np.delete(ys, indexes)
    # calculating the square error on the test sets
    l_s_cub = least_squares_cubic(xs, ys)
    arrangedYForTest_cub = l_s_cub[0] + l_s_cub[1]*testxs + l_s_cub[2]*(testxs ** 2) + l_s_cub[3]*(testxs **3)
    l_s_lin = least_squares_lin(xs, ys)
    arrangedYForTest_lin = l_s_lin[0] + l_s_lin[1]*testxs
    c_v_e_lin = squareError(testys, arrangedYForTest_lin)
    c_v_e_cub = squareError(testys, arrangedYForTest_cub)
    
    return c_v_e_lin, c_v_e_cub



# decides which line to use based of squareError ATM
def decider(tempxs, tempys, i):
    l_s_lin = least_squares_lin(tempxs, tempys)
    arrangedY_lin = l_s_lin[0] + l_s_lin[1]*tempxs
    l_s_cub = least_squares_cubic(tempxs, tempys)
    arrangedY_cub = l_s_cub[0] + l_s_cub[1] * tempxs + l_s_cub[2] * (tempxs ** 2) + l_s_cub[3]*(tempxs **3)
    c_v_e_lin, c_v_e_cub = crossValidation(tempxs,tempys)
    print("Linear cross validation error: ",c_v_e_lin)
    print("Cubic cross validation error: ",c_v_e_cub)
    if (c_v_e_cub < c_v_e_lin):
        print("This segment ",(i+1)," is cubic\n")
        return arrangedY_cub
    else:
        print("This segment ",(i+1)," is linear\n")
        return arrangedY_lin