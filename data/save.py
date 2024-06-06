import os
import random
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#given function for loading points from file
def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values
#given function for viewing data
def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

#General functions:
# 
#splits the data into an array of arrays of 20
def split(xarray, yarray, n):
    newx = xarray.reshape(n,20)
    newy = yarray.reshape(n,20)
    return newx, newy
# calculates the squared error
def squareError (actual_ys, estimated_ys):
    return np.sum((actual_ys - estimated_ys) ** 2)
# does the equation for calculating the coefficients
def equation_thing(x_e, y):
    return np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)


# least squares functions:
# 
# does the least squares thing for linear
def least_squares_lin(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = equation_thing(x_e, y)
    return v
# does the least squares thing for cubic
def least_squares_cubic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x, (x**2), (x**3)))
    v = equation_thing(x_e, y)
    return v
# does the least squares thing for unknown function
def least_squares_unk(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    v = equation_thing(x_e, y)
    return v


# decides which line to use based of squareError ATM
def decider(tempxs, tempys, i):
    l_s_lin = least_squares_lin(tempxs, tempys)
    arrangedY_lin = l_s_lin[0] + l_s_lin[1]*tempxs
    l_s_cub = least_squares_cubic(tempxs, tempys)
    arrangedY_cub = l_s_cub[0] + l_s_cub[1]*tempxs + l_s_cub[2]*(tempxs ** 2) + l_s_cub[3]*(tempxs **3)
    l_s_unk = least_squares_unk(tempxs,tempys)
    arrangedY_unk = l_s_unk[0] + l_s_unk[1]*(np.sin(tempxs))
    c_v_e_lin, c_v_e_cub, c_v_e_unk = k_foldCV(tempxs, tempys)
    print("Linear cross validation error: ",c_v_e_lin)
    print("Cubic cross validation error: ",c_v_e_cub)
    print("Unknown cross validation error: ",c_v_e_unk)
    if (c_v_e_unk < c_v_e_cub):
        print("segment ",(i+1),"is the unknown function (sin)\n")
        return arrangedY_unk
    if (c_v_e_cub  < c_v_e_lin):
        print("Segment ",(i+1)," is cubic\n")
        return arrangedY_cub
    else:
        print("Segment ",(i+1)," is linear\n")
        return arrangedY_lin
    

def crossValidation (xs, ys):
    # this bit divides the xs and ys array into two sepperate groups for testing 
    indexes = []
    ammountOfSamplesInTest = 5
    indexes = random.sample(range(0,19), ammountOfSamplesInTest)
    #these are the two testing arrays of length 5   
    testxs = np.array(xs[indexes])
    testys = np.array(ys[indexes])
    # taking away the elements in the test set from the whole set
    trainxs = np.delete(xs, indexes)
    trainys = np.delete(ys, indexes)
    # calculating the square error on the test sets
    l_s_testCub = least_squares_cubic(trainxs, trainys)
    l_s_testLin = least_squares_lin(trainxs, trainys)
    l_s_testUnk = least_squares_unk(trainxs, trainys)
    arrangedYForTest_cub = l_s_testCub[0] + l_s_testCub[1]*testxs + l_s_testCub[2]*(testxs ** 2) + l_s_testCub[3]*(testxs **3)
    arrangedYForTest_lin = l_s_testLin[0] + l_s_testLin[1]*testxs
    arrangedYForTest_unk = l_s_testUnk[0] + l_s_testUnk[1]*(np.sin(testxs))
    c_v_e_lin = squareError(testys, arrangedYForTest_lin)
    c_v_e_cub = squareError(testys, arrangedYForTest_cub)
    c_v_e_unk = squareError(testys, arrangedYForTest_unk)
    
    return c_v_e_lin, c_v_e_cub, c_v_e_unk



def k_foldCV(xs,ys):
    ammountAveragedOver = 100
    totalErrorLin = np.array([])
    totalErrorCub = np.array([])
    totalErrorUnk = np.array([])
    for i in range(ammountAveragedOver):
        errorLin, errorCub, errorUnk = crossValidation(xs,ys)
        totalErrorLin = np.append(totalErrorLin, errorLin)
        totalErrorCub = np.append(totalErrorCub, errorCub)
        totalErrorUnk = np.append(totalErrorUnk, errorUnk)
    averageErrorLin = sum(totalErrorLin)/ammountAveragedOver
    averageErrorCub = sum(totalErrorCub)/ammountAveragedOver
    averageErrorUnk = sum(totalErrorUnk)/ammountAveragedOver
    return averageErrorLin, averageErrorCub, averageErrorUnk



# main plotting function used
def plotMain(xs,ys, n):
    fig, ax = plt.subplots()
    for i in range(n):
        tempxs = xs[i]
        tempys = ys[i]
        arrangedY = decider(tempxs, tempys, i)
        ax.scatter(tempxs, tempys, s=30)
        ax.plot(tempxs, arrangedY, 'r-', lw=2)
    plt.show()

    return



def main():
    xs, ys = load_points_from_file("adv_3.csv")
    numberOfLines = xs.size // 20
    assert (numberOfLines == (ys.size//20))
    newx, newy = split(xs,ys,numberOfLines)
    plotMain(newx,newy, numberOfLines)



main()


if (c_v_e_unk < c_v_e_cub):
        print("segment ",(i+1),"is the unknown function (sin)\n")
        return arrangedY_unk
    if (c_v_e_cub  < c_v_e_lin):
        print("Segment ",(i+1)," is cubic\n")
        return arrangedY_cub
    else:
        print("Segment ",(i+1)," is linear\n")
        return arrangedY_lin