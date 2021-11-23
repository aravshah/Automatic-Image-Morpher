import os
from flask import Flask, render_template
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import sys, re, json
from datetime import datetime

import matplotlib
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import math
import pandas as pd

import torch

from scipy.spatial import Delaunay
from scipy.spatial import tsearch
import imageio

import multiprocessing
import dask

# Create the website object
app = Flask(__name__)

# App Global Variables
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output_images"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
sample_im1 = "static/ex_imgs/person1.jpg"
sample_im2 = "static/ex_imgs/person2.jpg"
sample_output = "static/ex_imgs/morph_person1_and_person2.gif"

# Global Variables
CORNERS = [[0,0], [0, 223], [223, 0], [223, 223]]
CORNERS = np.array(CORNERS)
FRAMES = 45

def load_model_from_file():
	model_name = "models/model_20_epochs.pt"
	model = torch.load(model_name, map_location=torch.device('cpu'))
	return model

def process_img(im):
    temp_im = im.astype(np.float32) / 255 - 0.5
    temp_im_c = resize(temp_im, (224, 224))
    temp_im_g = rgb2gray(temp_im_c)
    return temp_im_g, temp_im_c 


# Methods used for the morph function
def computeAffine(tri1_pts, tri2_pts):
    A = np.matrix("{} {} 1 0 0 0;".format(tri1_pts[0][0], tri1_pts[0][1])
                 +"0 0 0 {} {} 1;".format(tri1_pts[0][0], tri1_pts[0][1])
                 +"{} {} 1 0 0 0;".format(tri1_pts[1][0], tri1_pts[1][1])
                 +"0 0 0 {} {} 1;".format(tri1_pts[1][0], tri1_pts[1][1])
                 +"{} {} 1 0 0 0;".format(tri1_pts[2][0], tri1_pts[2][1])
                 +"0 0 0 {} {} 1".format(tri1_pts[2][0], tri1_pts[2][1]))

    b = np.matrix("{} {} {} {} {} {}".format(tri2_pts[0][0], tri2_pts[0][1], tri2_pts[1][0], tri2_pts[1][1], tri2_pts[2][0], tri2_pts[2][1]))
    b = np.transpose(b)
    
    result = np.vstack((np.reshape(np.linalg.lstsq(A, b, rcond=None)[0], (2, 3)), [0, 0, 1]))
    return result

def findAffine(tri, im_points, mid_points):
    # result = []
    # for x in tri.simplices:
    #     result.append(computeAffine(im_points[x, ], mid_points[x, ]))
    # return result

    # List Comprehension for slightly faster computation
    return [computeAffine(im_points[x, ], mid_points[x, ]) for x in tri.simplices]

def findMidWayFace(im1, im2, im1_points, im2_points, mid_points, tri, w):
    im1_affine_matrices = findAffine(tri, im1_points, mid_points)
    im2_affine_matrices = findAffine(tri, im2_points, mid_points)
    
    out_im = np.ones(im1.shape)
    
    for y in range(im1.shape[0]):
        for x in range(im1.shape[1]):
            i = tsearch(tri, (x, y))
            im1_affined_pts = np.dot(np.linalg.inv(im1_affine_matrices[i]), [x, y, 1])
            im2_affined_pts = np.dot(np.linalg.inv(im2_affine_matrices[i]), [x, y, 1])

            temp1 = im1[np.int64(im1_affined_pts[0, 1]), np.int64(im1_affined_pts[0, 0]), :] * w
            temp2 = im2[np.int64(im2_affined_pts[0, 1]), np.int64(im2_affined_pts[0, 0]), :] * (1-w)
            out_im[y, x, :] = temp1 + temp2
            
    return out_im

def findAverageShape(im1_points, im2_points, w):
    # ave_shape = []
    # for i in range(len(im1_points)):
    #     ave_shape.append([w * im1_points[i][0] + (1 - w) * im2_points[i][0], w * im1_points[i][1] + (1 - w) * im2_points[i][1]])
    # return np.array(ave_shape)

    # List Comprehension for slightly faster computation
    return np.array([[w * im1_points[i][0] + (1 - w) * im2_points[i][0], w * im1_points[i][1] + (1 - w) * im2_points[i][1]] for i in range(len(im1_points))])

# Moprhing parallelized
def morph_parallel(i, im1, im2, im1_points, im2_points):
    weights = np.linspace(0.0, 1.0, FRAMES)
    mid_Points = findAverageShape(im1_points, im2_points, weights[i])
    frame = findMidWayFace(im1, im2, im1_points, im2_points, mid_Points, Delaunay(mid_Points), weights[i])
    final_frame = (frame + 0.5) * 255
    res = final_frame.astype(np.uint8)
    return res


# Basic check to allow only image inputs
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define the view for the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index.html', my_im1=sample_im1, my_im2=sample_im2, my_out=sample_output)
    else:
        # check if the post request has the file part
        if 'file1' not in request.files:
            flash('No file1 part')
            return redirect(request.url)
        file1 = request.files['file1']
        if 'file2' not in request.files:
            flash('No file2 part')
            return redirect(request.url)
        file2 = request.files['file2']
        # if user does not select file or uploads without file
        if file1.filename == '':
            flash('No selected file1')
            return redirect(request.url)
        if file2.filename == '':
            flash('No selected file2')
            return redirect(request.url)
        # if file is not an image file
        if (not allowed_file(file1.filename)) or (not allowed_file(file2.filename)):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        # When the user uploads a file with good parameters
        filename1 = secure_filename(file1.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        filename2 = secure_filename(file2.filename)
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
        gif_length = request.form['length']
        return redirect(url_for('uploaded_file', filename1=filename1, filename2=filename2, gif_length=gif_length))

    
@app.route('/uploads/<filename1>_and_<filename2>_<gif_length>')
def uploaded_file(filename1, filename2, gif_length):

    img_1 = plt.imread(UPLOAD_FOLDER+"/"+filename1)
    img_2 = plt.imread(UPLOAD_FOLDER+"/"+filename2)
    model = app.config['MODEL']

    temp_im1, im1 = process_img(img_1)
    images = torch.from_numpy(temp_im1)
    images = torch.unsqueeze(torch.unsqueeze(images, 0), 0).float()
    outputs = model(images).float()
    pts = torch.reshape(outputs, shape = (68, 2)).detach().numpy()
    pts[:,0] *= 224
    pts[:,1] *= 224
    im1_pts = np.concatenate((pts, CORNERS), axis = 0)

    temp_im2, im2 = process_img(img_2)
    images = torch.from_numpy(temp_im2)
    images = torch.unsqueeze(torch.unsqueeze(images, 0), 0).float()
    outputs = model(images).float()
    pts = torch.reshape(outputs, shape = (68, 2)).detach().numpy()
    pts[:,0] *= 224
    pts[:,1] *= 224
    im2_pts = np.concatenate((pts, CORNERS), axis = 0)

    output_frames = []
    for i in range(FRAMES):
      temp = dask.delayed(morph_parallel)(i, im2, im1, im2_pts, im1_pts)
      output_frames.append(temp)

    fframes = dask.compute(output_frames)
    temp_fn1 = filename1.rsplit('.', 1)[0]
    temp_fn2 = filename2.rsplit('.', 1)[0]

    # adding local time to filename as a work around for flash caching.
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    exportname = OUTPUT_FOLDER+"/"+temp_fn1+"_and_"+temp_fn2+"_"+current_time+".gif"
    imageio.mimsave(exportname, fframes[0], format='GIF', duration= float(gif_length)/FRAMES)

    sample_im1_u = "/../" + sample_im1
    sample_im2_u = "/../" + sample_im2
    sample_output_u = "/../" + sample_output

    answer = "/../" + exportname
    im1_name =  "/../" + UPLOAD_FOLDER + "/" + filename1
    im2_name =  "/../" + UPLOAD_FOLDER + "/" + filename2
    final_answer = (im1_name, im2_name, answer)
    results.append(final_answer)
    return render_template('index.html', my_im1=sample_im1_u, my_im2=sample_im2_u, my_out=sample_output_u, results=results)

def main():
    model = load_model_from_file()
    
    app.config['SECRET_KEY'] = 'super secret key'
    app.config['MODEL'] = model
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run()

# Create a running list of results
results = []

#Launch everything
main()