import numpy as np
from flask import Flask, request
from PIL import Image
from flask_cors import CORS
from sklearn import datasets, svm, metrics
from joblib import load



# import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s], []))-set([s]))
             for s in squares)

classifier = load('model.joblib') 

def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    # To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  # (Fail if we can't assign d to square s.)
    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  # Already eliminated
    values[s] = values[s].replace(d, '')
    # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  # Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    # (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  # Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values

def solve(grid): 
    return search(parse_grid(grid))

def search(values):
    "Using depth-first search and propagation, try all possible values."
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in squares): 
        return values ## Solved!
    ## Chose the unfilled square s with the fewest possibilities
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d)) 
        for d in values[s])

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False


def get_start_x_point(frm, to, increment, y, data, threshold):
    for i in range(frm, to, increment):
        if(data[i, y] < threshold):
            return i


def get_start_y_point(frm, to, increment, x, data, threshold):
    for i in range(frm, to, increment):
        if(data[x, i] < threshold):
            return i


def hide_image(im1, start_x, start_y, end_x, end_y):
    im = im1.copy()
    width, height = im.size
    data = im.load()
    for y in range(height):
        for x in range(width):
            if(not (start_x < x < end_x and start_y < y < end_y)):
                data[x, y] = (255, 255, 255)
    return im


def processImageToArr(im, imSize=8):
    return 16 - (np.asarray(im.convert('L').resize(size=(imSize,imSize))).reshape(imSize*imSize)/16)


def image_to_number(im):
    arr = processImageToArr(im)
    newArr = []
    newArr.append(arr)
    newArr = np.asarray(newArr)
    return classifier.predict(newArr)[0]


def decode_image(im):
    width, height = im.size
    grayscaleImage = im.convert('L')
    data = grayscaleImage.load()
    threshold = 128
    start_x = get_start_x_point(0, width, 1, height/2, data, threshold)
    end_x = get_start_x_point(width-1, 0, -1, height/2, data, threshold)
    start_y = get_start_y_point(0, height, 1, width/2, data, threshold)
    end_y = get_start_y_point(height-1, 0, -1, width/2, data, threshold)
    if(not (-10<((end_x-start_x) - (end_y-start_y))<10)):
        print((end_x-start_x) - (end_y-start_y))
        raise Exception("Please crop properly")
    # num = 90
    ret = {}
    counter = 0
    im = im.crop((start_x, start_y, end_x, end_y))
    n_width, n_height = im.size
    for y in range(1, 10):
        for x in range(1, 10):
            # ret[squares[counter]] = ""+image_to_string(im.crop((start_x+((x-1)*end_x/9), start_y+((y-1)*end_y/9), x*end_x/9, y*end_y/9)))
            # ret[squares[counter]] = ""+image_to_string(hide_image(im, start_x+((x-1)*end_x/9), start_y+((y-1)*end_y/9), x*end_x/9, y*end_y/9))
            ret[squares[counter]] = str(image_to_number(im.crop(((x-1)*n_width/9, (y-1)*n_height/9, (x*n_width/9), (y*n_height/9)))))
            counter = counter+1
    return ret


def save_images(im, jsonValue):
    width, height = im.size
    grayscaleImage = im.convert('L')
    data = grayscaleImage.load()
    threshold = 128
    start_x = get_start_x_point(0, width, 1, height/2, data, threshold)
    end_x = get_start_x_point(width-1, 0, -1, height/2, data, threshold)
    start_y = get_start_y_point(0, height, 1, width/2, data, threshold)
    end_y = get_start_y_point(height-1, 0, -1, width/2, data, threshold)
    if(not (-10<((end_x-start_x) - (end_y-start_y))<10)):
        print((end_x-start_x) - (end_y-start_y))
        raise Exception("Please crop properly")
    # num = 90
    ret = {}
    counter = 0
    im = im.crop((start_x, start_y, end_x, end_y))
    n_width, n_height = im.size
    for y in range(1, 10):
        for x in range(1, 10):
            # ret[squares[counter]] = ""+image_to_string(im.crop((start_x+((x-1)*end_x/9), start_y+((y-1)*end_y/9), x*end_x/9, y*end_y/9)))
            # ret[squares[counter]] = ""+image_to_string(hide_image(im, start_x+((x-1)*end_x/9), start_y+((y-1)*end_y/9), x*end_x/9, y*end_y/9))
            ret[squares[counter]] = im.crop(((x-1)*n_width/9, (y-1)*n_height/9, (x*n_width/9), (y*n_height/9)))
            counter = counter+1
    for k in ret:
        ret[k].save("F:/Practice/python-programming/ml/datasets/sudoku/sudoku_dataset/"+str(jsonValue[k])+"/"+k+"B.jpg")

@app.route('/uploader', methods=['GET', 'POST'])
def file_uploader():
    if request.method == 'POST':
        f = request.files['file']
        im = Image.open(f)
        return decode_image(im)

def convert_to_string(grid):
    ret = ""
    for x in grid:
        ret += grid[x]
    # print(ret)
    return ret

@app.route('/solve', methods=['GET', 'POST'])
def solve_method():
    if request.method == 'POST':
        # print(request.json.get('body'))
        return solve(convert_to_string(request.json.get('body')))


# @app.route('/image-save-uploader', methods=['GET', 'POST'])
# def test_uploader():
#     if request.method == 'POST':
#         f = request.files['file']
#         jsonValue = request.form['json']
#         im = Image.open(f)
#         # save_images(im, json.loads(jsonValue))
#         return "Success"


if __name__ == '__main__':
    # app.run(port=3000,debug=True)
    app.run(debug=True)
