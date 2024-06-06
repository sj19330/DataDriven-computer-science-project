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
# given function for viewing data
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
    h = np.sum( (actual_ys - estimated_ys) ** 2)
    return h
# does the equation for calculating the coefficients
def equation_thing(extendedxs, y):
    h = np.linalg.inv(extendedxs.T.dot(extendedxs)).dot(extendedxs.T).dot(y)
    return h


# least squares functions:
# 
# does the least squares thing for linear
def least_squares_lin(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, x))
    v = equation_thing(extendedXs, y)
    return v
# does the least squares thing for cubic
def least_squares_cubic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, x, (x**2), (x**3)))
    v = equation_thing(extendedXs, y)
    return v
# does the least squares thing for unknown function
def least_squares_unk(x, y):
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, np.sin(x)))
    v = equation_thing(extendedXs, y)
    return v


# decides which line to use based of squareError ATM
def decider(tempxs, tempys, i):
    maxX = tempxs[19]
    minX = tempxs[0]
    spacedXs = np.linspace(minX, maxX, num=100, endpoint=True)
    c_v_e_lin, c_v_e_cub, c_v_e_unk = k_foldCV(tempxs, tempys)
    if (c_v_e_lin/1.2 < c_v_e_cub) and (c_v_e_lin/1.2 < c_v_e_unk):
        l_s_lin = least_squares_lin(tempxs, tempys)
        final = l_s_lin[0] + l_s_lin[1]*spacedXs
        decision = 'linear'
    elif (c_v_e_cub < c_v_e_unk):
        l_s_cub = least_squares_cubic(tempxs, tempys) 
        final = l_s_cub[0] + l_s_cub[1]*spacedXs + l_s_cub[2]*(spacedXs ** 2) + l_s_cub[3]*(spacedXs **3)
        decision = 'cubic'
    else: 
        l_s_unk = least_squares_unk(tempxs,tempys)
        final = l_s_unk[0] + l_s_unk[1]*(np.sin(spacedXs))
        decision = 'sine'
    return final, spacedXs, decision
        
    
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


def crossValidation (xs, ys):
    # this bit divides the xs and ys array into two sepperate groups for testing
    indexes = []
    ammountOfSamplesInTest = 5
    indexes = random.sample(range(0,20), ammountOfSamplesInTest)
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





# gets total error from that segment
def getErrorForSegment(decision, xs, ys):
    if decision == 'linear':
        l_s_lin = least_squares_lin(xs, ys)
        arrangedYLin = l_s_lin[0] + l_s_lin[1]*xs
        return squareError(ys, arrangedYLin)
    elif decision == 'cubic':
        l_s_cub = least_squares_cubic(xs, ys)
        arrangedYCub = l_s_cub[0] + l_s_cub[1]*xs + l_s_cub[2]*(xs**2) + l_s_cub[3]*(xs**3)
        return squareError(ys, arrangedYCub)
    elif decision == 'sine':
        l_s_sin = least_squares_unk(xs, ys)
        arrangedYSin = l_s_sin[0] + l_s_sin[1]*(np.sin(xs))
        return squareError(ys, arrangedYSin)

# main plotting function used
def plotMain(xs,ys, n):
    errors = np.array([])
    fig, ax = plt.subplots()
    for i in range(n):
        tempxs = xs[i]
        tempys = ys[i]
        arrangedY, spacedXs, decision = decider(tempxs, tempys, i)
        sq_error = getErrorForSegment(decision, tempxs, tempys)
        assert(sq_error != None)
        errors = np.append(errors, sq_error)
        ax.scatter(tempxs, tempys, s=30, c='#1f77b4')
        ax.plot(spacedXs, arrangedY, 'r-', lw=2)
    print(np.sum(errors))
    plt.show()
    return

def showMain(xs,ys, n):
    errors = np.array([])
    for i in range(n):
        tempxs = xs[i]
        tempys = ys[i]
        arrangedY, spacedXs, decision = decider(tempxs, tempys, i)
        sq_error = getErrorForSegment(decision, tempxs, tempys)
        assert(sq_error != None)
        errors = np.append(errors, sq_error)
    print(np.sum(errors))
    return



def main():
    if len(sys.argv) == 1:
        print("please give a datafile as an argument")
    elif len(sys.argv) == 2:
        xs, ys = load_points_from_file(sys.argv[1])
        numberOfLines = xs.size // 20
        assert (numberOfLines == (ys.size//20))
        newx, newy = split(xs,ys,numberOfLines)
        showMain(newx,newy, numberOfLines)
    elif len(sys.argv) == 3:
        x = '--plot'
        y = sys.argv[2]
        if y == x:
            xs, ys = load_points_from_file(sys.argv[1])
            numberOfLines = xs.size // 20
            assert (numberOfLines == (ys.size//20))
            newx, newy = split(xs,ys,numberOfLines)
            plotMain(newx,newy, numberOfLines)
        else:
            print("use argument '--plot' to see a visulisation")
    else:
        print("too many arguments")

main()