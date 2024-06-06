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
def least_squares_quadratic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, x, (x**2)))
    v = equation_thing(extendedXs, y)
    return v

# does the least squares thing for cubic
def least_squares_cubic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, x, (x**2), (x**3)))
    v = equation_thing(extendedXs, y)
    return v
# does the least squares thing for cubic
def least_squares_quartic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, x, (x**2), (x**3), (x**4)))
    v = equation_thing(extendedXs, y)
    return v
# does the least squares thing for cubic
def least_squares_quintec(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, x, (x**2), (x**3), (x**4), (x**5)))
    v = equation_thing(extendedXs, y)
    return v
# does the least squares thing for sine function
def least_squares_unk(x, y):
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, np.sin(x)))
    v = equation_thing(extendedXs, y)
    return v
# does the least squares thing for cos function
def least_squares_cos(x, y):
    ones = np.ones(x.shape)
    extendedXs = np.column_stack((ones, np.cos(x)))
    v = equation_thing(extendedXs, y)
    return v


# decides which line to use based of squareError ATM
def decider(tempxs, tempys, i):
    maxX = tempxs[19]
    minX = tempxs[0]
    # spacedXs = np.linspace(minX, maxX, num=100, endpoint=True)
    spacedXs = tempxs
    l_s = least_squares_quadratic(tempxs, tempys)
    final = l_s[0] + l_s[1]*tempxs + l_s[2]*(tempxs**2)
    decision = 'quadratic'
    # l_s = least_squares_quartic(tempxs, tempys)
    # final = l_s[0] + l_s[1]*spacedXs + l_s[2]*(spacedXs**2) + l_s[3]*(spacedXs**3) + l_s[4]*(spacedXs**4)
    # decision = 'quartic'
    # l_s = least_squares_quintec(tempxs, tempys)
    # final = l_s[0] + l_s[1]*spacedXs + l_s[2]*(spacedXs**2) + l_s[3]*(spacedXs**3) + l_s[4]*(spacedXs**4) +l_s[5]*(spacedXs**5)
    # decision = 'quintec'
    # if (c_v_e_lin/1.2 < c_v_e_cub) and (c_v_e_lin/1.2 < c_v_e_unk):
    # l_s_lin = least_squares_lin(tempxs, tempys)
    # final = l_s_lin[0] + l_s_lin[1]*spacedXs
    # decision = 'linear'
    # elif (c_v_e_cub < c_v_e_unk):
    # l_s_cub = least_squares_cubic(tempxs, tempys) 
    # final = l_s_cub[0] + l_s_cub[1]*spacedXs + l_s_cub[2]*(spacedXs ** 2) + l_s_cub[3]*(spacedXs **3)
    # decision = 'cubic'
    # else: 
    # l_s_unk = least_squares_unk(tempxs,tempys)
    # final = l_s_unk[0] + l_s_unk[1]*(np.sin(spacedXs))
    # decision = 'sine'
    # l_s = least_squares_cos(tempxs,tempys)
    # final = l_s[0] + l_s[1]*(np.cos(spacedXs))
    # decision = 'cos'
    return final, spacedXs, decision
        
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
    elif decision == 'quadratic':
        l_s_quad = least_squares_quadratic(xs, ys)
        arrangedYQuad = l_s_quad[0] + l_s_quad[1]*xs + l_s_quad[2]*(xs**2)
        return squareError(ys, arrangedYQuad)
    elif decision == 'quartic':
        l_s_quart = least_squares_quartic(xs, ys)
        arrangedYQuar = l_s_quart[0] + l_s_quart[1]*xs + l_s_quart[2]*(xs**2) + l_s_quart[3]*(xs**3) + l_s_quart[4]*(xs**4)
        return squareError(ys, arrangedYQuar)
    elif decision == 'quintec' :
        l_s_quin = least_squares_quintec(xs, ys)
        arrangedYQuin = l_s_quin[0] + l_s_quin[1]*xs + l_s_quin[2]*(xs**2) + l_s_quin[3]*(xs**3) + l_s_quin[4]*(xs**4) +l_s_quin[5]*(xs**5)
        return squareError(ys, arrangedYQuin)
    elif decision == 'cos' :
        l_s_cos = least_squares_cos(xs, ys)
        arrangedYCos = l_s_cos[0] + l_s_cos[1]*(np.cos(xs))
        return squareError(ys, arrangedYCos)
 
# main plotting function used
def plotMain(xs,ys, n):
    errors = np.array([])
    fig, ax = plt.subplots()
    for i in range(n):
        tempxs = xs[i]
        tempys = ys[i]
        arrangedY, spacedXs, decision = decider(tempxs, tempys, i)
        sq_error = getErrorForSegment(decision, tempxs, tempys)
        print(sq_error)
        assert(sq_error != None)
        errors = np.append(errors, sq_error)
        ax.scatter(tempxs, tempys, s=30, c='#1f77b4')
        ax.plot(spacedXs, arrangedY, 'r-', lw=2)
    print(np.sum(errors)/len(errors))
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