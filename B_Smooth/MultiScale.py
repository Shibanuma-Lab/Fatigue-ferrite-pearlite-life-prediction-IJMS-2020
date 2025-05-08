import os  # The os library is used to set up the working directory
import re  # Define regular expression patterns
import math # The math library is used to perform mathematical operations
import json # Use the built-in JSON-related functions in Mathematica to save the data in JSON format, and then use JSON libraries in Python to process the JSON data
import random # The random library is used to generate random numbers
import pickle # The pickle library is used to save and read data in pkl format
import numpy as np  # The NumPy library is used to process arrays
import pandas as pd  # Using the pandas library to read CSV files
import datetime # The datetime library is used to record the time of execution
from scipy import  interpolate # The scipy library is used to perform interpolation
from scipy.interpolate import griddata # The scipy library is used to perform interpolation
from scipy.interpolate import interp1d # The scipy library is used to perform interpolation
from scipy.interpolate import LinearNDInterpolator # The scipy library is used to perform interpolation
from scipy.optimize import brentq
from itertools import groupby
import warnings 
warnings.filterwarnings("ignore", category=np.RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from scipy.interpolate import RBFInterpolator


###############################################STRUCTURE OF THE MODEL###############################################
# 1. Basic information import (class: BasicParameters)
# 2. ABAQUS data import (class: AbaqusDatabaseCreator)
# 3. Material data import (class: MaterialDataImporter)
# 4. Creat Field value function (class: FieldValuesFunction)
# 5. Life evaluation for single crack (fuction: CrackLifeCalc)
# 6. Life evaluation for sigle area element (fuction: ElementLife)
# 7. Execution of the model
###############################################STRUCTURE OF THE MODEL###############################################





# -------------------------------------------------------------------------------------------------------------------------------------------  
# ----------------------------------------------------------1. Basic information import--------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------  

class BasicParameters:
    def __init__(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Initialize all parameters
        self.name = os.path.basename(os.getcwd()) # The name of the specimen
        self.steel = self.name.split("_")[:3][0] # The steel type of the specimen
        self.type = self.name.split("_")[:3][1] # The type of the test
        self.T = self.name.split("_")[:3][2] # Thickness sign
        self.crystal = "BCC" # The crystal structure of the material
        self.n_symm = 4 #Number of model calculations (typically 4 for 1/4 model)
        self.iteration_num = np.arange(0,1,1) #iteration times
        self.FirstGrain = 'Ferrite' #first grain type
        self.SecondGrain = 'Pearlite' #second grain type

        # Parameters related to fissure aperture ratio
        self.closure_type = 2 # The closure type of the specimen


        #life calculation
        self.c_paris = 1 #Paris'law coefficient
        filename = os.path.basename(__file__)
        matches = re.findall(r"([A-Za-z])(-?\d+(?:\.\d+)?)", filename)
        if matches:
            try:
                minS = float([m[1] for m in matches if m[0] == 'L'][0])
                maxS = float([m[1] for m in matches if m[0] == 'U'][0])
                Sstep = float([m[1] for m in matches if m[0] == 'S'][0])
                self.r = float([m[1] for m in matches if m[0] == 'R'][0])
                print(minS, maxS, Sstep, self.r)
            except IndexError:
                print("error: unable to extract all values")
        else:
            print("error: no matches found")
        
        self.σ_nom = np.arange(minS,maxS,Sstep)#Applied stress


        self.k_tr = 2 * 14 #Material constant for opening stress calculation
        self.gb_effect = 1 #Grain boundary effect type
        self.remote_type = 2
         #Remote stress conversion type

        #life calculation
        self.c_paris = 1 #Paris'law coefficient
        self.n_paris = 1.8 #Paris'law exponent
        self.Δδ_th = 0.000063 #threshold for crack-tip sliding displacement range

        #specimen dimentions
        self.width = None # The width of the specimen
        self.thickness = None # The thickness of the specimen

        #Active area defination
        self.y_lim = None # The y-coordinate of active zone range
        self.z_lim = None # The z-coordinate of active zone range
        self.grain_size_lim = None # Ratio of the smallest surface grain to the largest grain size for calculation
        self.active_element_size = None # The size of the active zone
        self.active_element_area = None # The area of the active zone
        self.stress_lim = None # Ratio of the smallest shear stress to the largest shear stress for calculation on area element
        self.AeNumY = None # The number of area elements in the y direction
        self.AeNumZ = None # The number of area elements in the z direction
        self.AeNum = None # The number of area elements in the target area
        self.AeYZ = None # YZ coordinates of the active zone

        #Evaluation points
        self.eval_num_stage1 = None #The number of evaluation points in Stage I (within first grains)
        self.eval_num_stage2 = None #The number of evaluation points in Stage II (outside second grains)
        self.eval_num_total = None #The total number of evaluation points
        self.eval_lenghth_lim = None #Evaluation limination of crack lenghth
        self.eval_points_full= None #Position of evaluation points(stage1+stage2)
        self.eval_points_stage1 = None #Position of evaluation points in Stage I (within first grains)
        self.eval_points_stage2 = None #Position of evaluation points in Stage II (outside the first grains)

  


        #call fuctions
        self.specimen_dimentions()
        self.active_zone()
        self.calculate_evaluation_points()
#------------------------------------------------------------------------------------------------------------------            
    #Specimen dimentions
    def specimen_dimentions(self):
        self.width = 5 if self.type in ["Smooth", "R008", "R100", "SmoothT" ] else 1.5
        self.thickness = {
            "T16":1.35,"T24": 1.9, "T25": 2.0, "T50": 4.5, "T65": 6.0, "T100": 9.5,
        }.get(self.T)

    #Active zone defination
    def active_zone(self):
        if self.type == "Smooth" or self.type == "SmoothT":
            self.y_lim = 1.5
            self.z_lim = 2.5
            self.grain_size_lim = 0.2
            self.active_element_size = [0.1, 0.1]
            self.stress_lim = 0.1

        elif self.type == "R100":
            self.y_lim = 0.5
            self.z_lim = 2.5
            if self.steel == "B" or self.steel == "E":
                self.grain_size_lim = 0.2
                self.active_element_size = [0.125, 0.125]
                self.stress_lim = 0.5
            elif self.steel == "N50R":
                self.grain_size_lim = 0.7
                self.active_element_size = [0.05, 0.125]
                self.stress_lim = 0.5
            elif self.steel == "Bainite":
                self.grain_size_lim = 0.6
                self.active_element_size = [0.05, 0.125]
                self.stress_lim = 0.5

        elif self.type == "R008":
            self.y_lim = 0.04
            self.z_lim = 2.5
            self.grain_size_lim = 0.2
            self.active_element_size = [0.02, 0.05]
            self.stress_lim = 0.2
        elif self.type == "CS":
            self.y_lim = 0.025
            self.z_lim = 0.75
            self.grain_size_lim = 0.2
            self.active_element_size = [0.025, 0.15]
            if self.steel in ["E", "G", "MM1", "MM2", "MM3", "LM2", "ML2", "MS2", "SM2", "SS2"]:
                self.stress_lim = 1.0
            elif self.steel == "N50R":
                self.stress_lim = 1.0
            elif self.steel == "G":
                self.stress_lim = 0.6
        elif self.type == "CSS":
            self.y_lim = 1.5
            self.z_lim = 0.75
            self.grain_size_lim = 0.1
            self.active_element_size = [0.1, 0.15]
            if self.steel in ["E", "G", "MM1", "MM2", "MM3", "LM2", "ML2", "MS2", "SM2", "SS2"]:
                self.stress_lim = 0.1

        self.active_element_area = self.active_element_size[0] * self.active_element_size[1]
        
        # Creation of area element coordinates
        self.AeNumY = int(self.y_lim / self.active_element_size[0])
        self.AeNumZ = int(self.z_lim / self.active_element_size[1])
        self.AeNum = self.AeNumY * self.AeNumZ  # Number of area elements in the target area
        self.AeYZ = np.array([[self.active_element_size[0] * (i - 0.5 if i > 0 else 0), 
                               self.active_element_size[1] * (j - 0.5 if j > 0 else 0)]
                      for i in range(1,self.AeNumY + 1)  # Start from 0 to include points with y-coordinate as 0
                      for j in range(1,self.AeNumZ + 1)])


    #Evaluation points
    def calculate_evaluation_points(self):
        if self.gb_effect == 0:
            self.eval_num_stage1= 4
            self.eval_num_stage2 = 8
            self.eval_lenghth_lim = 0.95 * self.thickness

            def eval_points_full(r1):
                return [r1 * (1 - ((self.eval_num_stage1 - i) / self.eval_num_stage1) ** 3) for i in range(self.eval_num_stage1)] + \
                       [r1 + (self.eval_lenghth_lim - r1) * ((i / self.eval_num_stage2) ** 1.2) for i in range(self.eval_num_stage2)]
            self.eval_points_full = eval_points_full
        else:
            self.eval_num_stage1 = 4
            self.eval_lenghth_lim = 0.98 * self.thickness
            def eval_points_stage1(r1):
                return [r1 * (1 - ((self.eval_num_stage1 - i + 1) / self.eval_num_stage1) ** 3) for i in range(1, self.eval_num_stage1 + 1)]
            self.eval_points_stage1 = eval_points_stage1


            if self.steel in ["E" ,"G", "MM1", "MM2", "MM3","LM2","ML2","MS2","SM2","SS2","B"]:
                if self.T == "T16":
                    self.eval_points_stage2 = {
                        "Smooth": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35]}.get(self.type)
                if self.T == "T24":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8], 
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8]}.get(self.type)
                elif self.T == "T25":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9]}.get(self.type)
                elif self.T == "T50":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4],
                        "CSS": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 
                                0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 
                                0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 
                                0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 
                                0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.05, 1.10, 1.15, 1.20,
                                1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.10, 2.15, 2.20, 2.25,
                                2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60, 2.65, 2.70, 2.75, 2.80, 2.85, 2.90, 2.95, 3.00, 3.05, 3.10, 3.15, 3.20, 3.25,
                                3.30, 3.35, 3.40, 3.45, 3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90, 3.95, 4.00, 4.05, 4.10, 4.15, 4.20, 4.25,
                                4.30, 4.35, 4.40]}.get(self.type)
                elif self.T == "T65":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9]}.get(self.type)
                elif self.T == "T100":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, \
                            7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, \
                            7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4]}.get(self.type)

            elif self.steel == "N50R":
                if self.T == "T24":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8]}.get(self.type)
                elif self.T == "T25":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9]}.get(self.type)
                elif self.T == "T50":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4],
                        # "CS": [0.05, 0.10, 0.15, 0.2,  0.3,  0.4,  0.6,  0.8, 1.0,  1.5, 2.0,  2.5,  3.0,  3.5, 4.0,  4.4]
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4]
                            }.get(self.type)
                elif self.T == "T65":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9]}.get(self.type)
                elif self.T == "T100":
                    self.eval_points_stage2 = {
                        "R008": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, \
                            7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4],
                        "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                            1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                            3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, \
                            5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, \
                            7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4]}.get(self.type)


        self.eval_num_stage2 = len(self.eval_points_stage2) + 2 
        self.eval_num_total = self.eval_num_stage1 + self.eval_num_stage2

    def SlipPlane(self, ang, DeltaSigma):#
        # ang: Euler angles of the grain
        # DeltaSigma: stress tensor range
        nv = {
            "BCC": np.array([
                    [[1., 1., 0.], [1., -1., 1.], [-1., 1., 1.]],
                    [[1., -1., 0.], [1., 1., 1.], [-1., -1., 1.]],
                    [[1., 0., 1.], [1., 1., -1.], [-1., 1., 1.]],
                    [[-1., 0., 1.], [-1., 1., -1.], [1., 1., 1.]],
                    [[0., 1., 1.], [1., 1., -1.], [1., -1., 1.]],
                    [[0., 1., -1.], [1., 1., 1.], [1., -1., -1.]]]),
            "FCC": np.array([
                    [[1., 1., 1.], [1., -1., 0.], [0., 1., -1.], [1., 0., -1.]],
                    [[-1., 1., 1.], [1., 1., 0.], [0., 1., -1.], [1., 0., 1.]],
                    [[1., -1., 1.], [1., 1., 0.], [0., 1., 1.], [1., 0., -1.]],
                    [[1., 1., -1.], [1., -1., 0.], [0., 1., 1.], [1., 0., 1.]]])
        }.get(self.crystal)
        #Define slip planes and directions for FCC structure
        C_Psi, C_Theta, C_Phi = np.cos(ang)
        S_Psi, S_Theta, S_Phi = np.sin(ang)

        g_Psi_Theta_Phi = np.array([                            # Rotation matrix
            [C_Psi*C_Theta*C_Phi - S_Psi*S_Phi, S_Psi*C_Theta*C_Phi + C_Psi*S_Phi, -S_Theta*C_Phi],
            [-C_Psi*C_Theta*S_Phi - S_Psi*C_Phi, -S_Psi*C_Theta*S_Phi + C_Psi*C_Phi, S_Theta*S_Phi],
            [C_Psi*S_Theta, S_Psi*S_Theta, C_Theta]
            ])

        nv2 = np.array([list(map(lambda x: g_Psi_Theta_Phi.dot(x), sublist)) for sublist in nv])# 
        # Compute the shear stress for each slip system

        tau_list = []
        num_plane, num_direction, _ = nv.shape
        for i in range(num_plane): # There are 4 slip plane in FCC
            for j in range(num_direction-1): # There are 3 slip direction in each slip plane
                tau = abs(np.dot(np.dot(nv2[i, 0], DeltaSigma), nv2[i, j + 1]))
                tau_list.append((tau, i, j))
        
        tau_list.sort(key=lambda x: x[0], reverse=True)
        i0, j0 = tau_list[0][1], tau_list[0][2]
        
        max_shear_stress = 1.0 / np.sqrt(6) * tau_list[0][0] 
        max_principal_stress_dir = 1.0 / np.sqrt(2) * np.sign(nv2[i0, 0, 0]) * nv2[i0, 0]
        max_theta_n = 1.0 / np.sqrt(3) * np.sign(nv2[i0, 0, 0]) * nv2[i0, j0 + 1]

        return max_shear_stress, np.array(max_principal_stress_dir), np.array(max_theta_n)
        
    def makeEulerAngles(self):#
        return ([random.uniform(0, 2 * np.pi), math.acos(random.uniform(-1, 1)), random.uniform(0, 2 * np.pi)])


#Part1 End
#
#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------2. ABAQUS data import ---------------------------------------------------------------            
# -------------------------------------------------------------------------------------------------------------------------------------------            

class AbaqusDatabaseCreator:
    def __init__(self, BasicParams):
        #From class BasicParameters, the parameters are imported into the AbaqusDataImporter class.
        self.BasicParams = BasicParams

        #Reading .inp files.
        self.inp_data = open(self.BasicParams.name + ".inp", encoding="iso-8859-1").readlines() #Read the .inp file and save it as a list of strings.
        #Reading .dat files.
        self.dat_data = open(self.BasicParams.name + ".dat", encoding="iso-8859-1").readlines() #Read the .dat file and save it as a list of strings.

        #Reading of material constants
        self.E = None #Young's modulus
        self.ν = None #Poisson's ratio
        self.AA = None #Parameters used in crack and grain boundary interaction theory

        #Reading model information
        self.node0 = None
        self.node = None 
        self.elements = None
        self.surface = None
        self.xyz = None
        self.nnm = None
        self.triming = None
        self.step = None
        self.sigma_nom_list = None
        self.f_sigma = None
        self.DataEES = None
        self.DataPES = None
        self.DataAES = None
        self.FDataE = None
        self.FDataP = None
        self.FDataA = None
        self.EES = None
        self.PES = None
        self.AES = None

        #call fuctions
        self.read_material_paramas()
        self.read_inp()
        self.read_dat()

#------------------------------------------------------------------------------------------------------------------            
    #Material parameters
    def read_material_paramas(self):
        elastic = [i for i, line in enumerate(self.inp_data) if "*Elastic" in line][0]
        values_line = self.inp_data[elastic + 1].split(",")
        self.E, self.ν = map(float, values_line)
        self.AA = self.E / (4 * np.pi * (1 - self.ν ** 2))

    #Model information
    def read_inp(self):
        #----------------------------------------------sub_function_START------------------------------------------------
        def move_instance(inst, xyz0): # Definition of the function for transforming the coordinates of nodes.
            tr = np.array(inst[0][:3]) 
            xyz1 =xyz0+tr
            a0 = np.array(inst[1][:3])   
            a1 = np.array(inst[1][3:6])  
            axis = (a1 - a0) / np.linalg.norm(a1 - a0) 
            phi = np.pi * inst[1][-1] / 180
            def rformula(xyz):
                r = xyz - a0
                return r * np.cos(phi) + axis * (np.dot(axis, r)) * (1 - np.cos(phi)) + np.cross(axis, r) * np.sin(phi)
            xyz2 = [list(map(rformula,  xyz1))]  
            return xyz2+a0
        
        def x0(y, z): 
            if y > y_max:
                return np.array([x_min_filter])
            else:
                interpolated_value = griddata(points, x, (y, z), method='linear')
                return interpolated_value  
        

        
        def active_zone_element(n): # Define the function to extract the area elements
            a_start = [idx for idx, line in enumerate(self.inp_data) if "elset=_ActiveZone_S" + str(n+1) in line][0] +1
            inp_a_start0 = list(self.inp_data[a_start].split(","))
            inp_a_start1 = [int(item) for item in inp_a_start0]

            if self.inp_data[a_start - 1].split(",")[-1] == ' generate\n':
                ae = list(range(inp_a_start1[0], inp_a_start1[1]+1, inp_a_start1[2]))
                print("ae", ae)
                
            else:
                a_end_index = l_symbol.index(a_start-1) + 1
                a_end = l_symbol[a_end_index]-1
                print("000")
                print("a_start", a_start, "a_end", a_end)
                ae_lines = self.inp_data[a_start:a_end + 1]
                ae_lines = [line.strip() for line in ae_lines]
                ae0 = [list(map(int, line.split(",")[0:16])) for line in ae_lines] 
                ae = [item for sublist in ae0 for item in sublist] 

            selected_elements_ActiveZone = [sublist for index, sublist in enumerate(self.elements) if index + 1 in ae] 
            ActiveZone_n0= [[sublist[i] for i in surface[n]] for sublist in selected_elements_ActiveZone]
            return [item for sublist in ActiveZone_n0 for item in sublist]
#----------------------------------------------sub_function_END------------------------------------------------
       
       
       
        l_node = [i for i, line in enumerate(self.inp_data) if "*Node" in line]
        l_node_set = [i for i, line in enumerate(self.inp_data) if "*Nset" in line]
        l_element = [i for i, line in enumerate(self.inp_data) if "*Element" in line]
        l_element_set = [i for i, line in enumerate(self.inp_data) if "*Elset" in line]
        l_instance = [i for i, line in enumerate(self.inp_data) if "*Instance" in line]
        l_surface = [i for i, line in enumerate(self.inp_data) if "*Surface" in line]
        l_symbol = sorted(set().union(l_node, l_node_set, l_element, l_element_set, l_instance, l_surface))

        #node
        node_start = l_node[0] + 1
        node_end = l_element[0] - 1
        node_lines = self.inp_data[node_start:node_end + 1] 
        self.node0 = np.array([list(map(float, line.split(",")[1:4])) for line in node_lines])

        #element
        element_start = l_element[0] + 1
        element_end = l_node_set[0] - 1
        element_lines = self.inp_data[element_start:element_end + 1]
        self.elements = np.array([list(map(int, line.split(",")[1:9])) for line in element_lines])

        #coordinate transformation
        instance_start_index = next((i + 1 for i, line in enumerate(self.inp_data) if "*Instance, name=Specimen, part=Specimen" in line), None)
        end_instance_index = next((i for i in range(instance_start_index, len(self.inp_data)) if "*End Instance" in self.inp_data[i]), None)
        instance_data_lines = [line.strip() for line in self.inp_data[instance_start_index:(end_instance_index)]]
        instance = [list(map(float, line.split(','))) for line in instance_data_lines]  
        instance.append(["*End Instance"])
        if instance[0] == ["*End Instance"]:
            instance = [[0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
        if instance[1] == ["*End Instance"]:
            instance = [instance[0], [1, 0, 0, 0, 0, 0, 0]]     
        self.node = np.round(move_instance(instance, self.node0)[0], 4)

        # Definition of the nodal set Triming in the Active Zone
        eTrimStart = [i for i, item in enumerate(self.inp_data) if "elset=Trim" in item][0] + 1
        if self.inp_data[eTrimStart - 1].split(',')[-1]==" generate\n": 

            eTrim0 = [list(range(int(self.inp_data[eTrimStart].split(',')[0]), int(self.inp_data[eTrimStart].split(',')[1]) + 1, int(self.inp_data[eTrimStart].split(',')[2])))]
            
            eTrim = [x - 1 for x in eTrim0]
        else:
            eTrimStart_index = l_symbol.index(eTrimStart-1) + 1        
            eTrimEnd = l_symbol[eTrimStart_index]-1
            eTrim_lines = self.inp_data[eTrimStart:eTrimEnd + 1]
            eTrim00 = [list(map(int, line.split(",")[0:16])) for line in eTrim_lines]
            eTrim0 = [item for sublist in eTrim00 for item in sublist]
            eTrim = [x - 1 for x in eTrim0]
        # while eTrim[-1] == 0:
        #     eTrim.pop()
        Triming0 =(np.unique(self.elements[eTrim, :].flatten())-1).tolist()   
        
        # Extract Z-coordinates of Active Zone       
        surface =(np.array( [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8]])-1).tolist()
        AN0 = []
        for n in range(6):
            if any("elset=_ActiveZone_S" + str(n+1) in item for item in self.inp_data):  
                AN0.append(active_zone_element(n))
            else:
                AN0.append([])
        AN = sorted(set(item-1 for sublist in AN0 if sublist for item in sublist))  

        # (Function defining the x-coordinate of the surface of the active zone as (y,z)*)
        selected_node = self.node[AN,:]
        x = selected_node[:,[0]]
        y = selected_node[:,[1]]
        z = selected_node[:,[2]]
        x_min_filter = np.min(x)
        y_max = np.max(y)

        points = np.column_stack((y, z))
        selected_node_Triming = self.node[Triming0,:]

        xyz0 =np.array([[coord[0] - x0(coord[1], coord[2])[0], coord[1], coord[2]] for coord in selected_node_Triming])
        ymax = np.max(1.2 *selected_node[:,[1]].flatten())
        zmax = np.max(1.2 *selected_node[:,[2]].flatten())
        condition1=(xyz0[:, 1] < ymax) & (xyz0[:, 2] < zmax)
        self.xyz = xyz0[condition1]
        self.nnm = len(self.xyz)
        self.triming = [idx for idx, point in enumerate(xyz0) if point[1] < ymax and point[2] < zmax]


    #read abaqus datebase
    def read_dat(self):
#-----------------------------------------------------------sub_function__START------------------------------------------------
        def σ_eqE(ee): # Functions returning elastic components
            return np.dot(CCC(0.3), ee)

        def σ_eqP(pe): # Functions returning plastic components
            return np.dot((CCC(0.45) + CCC(0.55)) / 2, pe)
        
        def CCC(nu_temp): # Definition of the elastic matrix function CCC
            GG = self.E / (2 * (1 + nu_temp))
            G0 = ((1 - nu_temp) * self.E) / ((1 + nu_temp) * (1 - 2 * nu_temp))
            G1 = (nu_temp * self.E) / ((1 + nu_temp) * (1 - 2 * nu_temp))
            
            return np.array([[G0, G1, G1, 0., 0., 0.],
                            [G1, G0, G1, 0., 0., 0.],
                            [G1, G1, G0, 0., 0., 0.],
                            [0., 0., 0., GG, 0., 0.],
                            [0., 0., 0., 0., GG, 0.],
                            [0., 0., 0., 0., 0., GG]])

        def MapThread(data1, data2): #pair
            data11=data1.tolist()
            data22=data2
            return [list(pair) for pair in zip(data11, data22)]    
#----------------------------------------------sub_function__END------------------------------------------------
  
  
        LocationMAXIMUM = [idx-2 for idx, line in enumerate(self.dat_data) if "MAXIMUM" in line]

        # Elastic strain tensor
        EE_start = [idx+3 for idx, line in enumerate(self.dat_data) if "EE11" in line]
        EE_end1 =list(map(lambda x: np.heaviside(x,0), list(map(lambda x: x-np.array(LocationMAXIMUM), EE_start))))
        EE_end2 =list(map(lambda x: int(np.sum(x))+1, EE_end1)) 
        EE_end = [sublist for index, sublist in enumerate(LocationMAXIMUM) if index + 1 in EE_end2]

        DataEE0_line0 = [self.dat_data[start:end+1] for start, end in zip(EE_start, EE_end)]
        DataEE0 = np.array(list(map(lambda line_i: [list(map(float, line.split()[1:])) for line in line_i], DataEE0_line0)))
        DataEE =list(map(lambda DataEE0_sub: DataEE0_sub[self.triming, :], DataEE0))
        DataEES0 = [list(map(σ_eqE, sublist)) for sublist in DataEE]

        # # Plastic strain tensor
        PE_start = [idx+3 for idx, line in enumerate(self.dat_data) if "PE11" in line]
        PE_end1 =list(map(lambda x: np.heaviside(x,0), list(map(lambda x: x-np.array(LocationMAXIMUM), PE_start))))
        PE_end2 =list(map(lambda x: int(np.sum(x))+1, PE_end1))
        PE_end = [sublist for index, sublist in enumerate(LocationMAXIMUM) if index + 1 in PE_end2]

        DataPE0_line0 = [self.dat_data[start:end+1] for start, end in zip(PE_start, PE_end)]
        DataPE0 = np.array(list(map(lambda line_i: [list(map(float, line.split()[1:7])) for line in line_i], DataPE0_line0)))
        DataPE = list(map(lambda DataPE0_sub: DataPE0_sub[self.triming, :], DataPE0))
        DataPES0 = [list(map(σ_eqP, sublist)) for sublist in DataPE]

        # Equivalent stress tensor
        DataS0 = [[0., 0., 0., 0., 0., 0.] for _ in range(self.nnm)]

        self.DataEES = [DataS0] + DataEES0
        self.DataPES = [DataS0] + DataPES0
        self.DataAES = (np.array(self.DataEES) + np.array(self.DataPES)).tolist()

        xyzDE = list(map(lambda DataEES_sub: MapThread(self.xyz,DataEES_sub), self.DataEES)) 
        xyzDP = list(map(lambda DataPES_sub: MapThread(self.xyz,DataPES_sub), self.DataPES)) 
        xyzDA = list(map(lambda DataAES_sub: MapThread(self.xyz,DataAES_sub), self.DataAES))
        self.step = len(xyzDE)

        if self.BasicParams.type == "Smooth":
            LoadLines = [index+ 1 for index, row in enumerate(self.inp_data) if '*Dsload' in row]
            sigma_app_edge = [float(self.inp_data[line].split(",")[2]) for line in LoadLines]

        
            if self.BasicParams.steel == "Bainite" or self.BasicParams.steel == "Martensite":
                sigma_nom_list0 = [abs((15 * 2) / (self.BasicParams.width * self.BasicParams.thickness) * σ) for σ in sigma_app_edge]
            else:
                sigma_nom_list0 = [(15 * 1.6) / (self.BasicParams.width * self.BasicParams.thickness) * σ for σ in sigma_app_edge]
        
        elif self.BasicParams.type == "R008" or self.BasicParams.type == "R100":
            print("type == R008 or R100", self.BasicParams.type)
            RF1Lines0 = [idx+4 for idx, line in enumerate(self.dat_data) if "RF1" in line]
            RF1Lines = [sublist for index, sublist in enumerate(self.dat_data) if index + 1 in RF1Lines0]
            Load = [4 * float(line.split()[1]) for line in RF1Lines]
            sigma_nom_list0 = [((50 * load0) / 4) / ((self.BasicParams.width * self.BasicParams.thickness**2) / 6) for load0 in Load]
            print()
        
        elif self.BasicParams.type in ["CS","CSS"]:
            Load_Lines0 = [idx+2 for idx, line in enumerate(self.inp_data) if "*Dsload" in line]
            Load_lines = [sublist for index, sublist in enumerate(self.inp_data) if index + 1 in Load_Lines0]
            sigma_app_edge = [-1 * float(line.split(",")[2]) for line in Load_lines]
            sigma_nom_list0 = [(self.BasicParams.width * (self.BasicParams.thickness+0.5)) / (self.BasicParams.width * self.BasicParams.thickness) * σ for σ in sigma_app_edge]

        elif self.BasicParams.type == "SmoothT":
            LoadLines = [index+ 1 for index, row in enumerate(self.inp_data) if '*Cload' in row]
            Force_app = [float(self.inp_data[line].split(",")[2]) for line in LoadLines]

        
            if self.BasicParams.steel == "H":
                sigma_nom_list0 = [F / (self.BasicParams.width * self.BasicParams.thickness/4) for F in Force_app]


        self.sigma_nom_list = [0.] + sigma_nom_list0[0:self.step - 1]
        

        self.FDataE = [[[[xyzDE[i][j][0], xyzDE[i][j][1][k]] for j in range(self.nnm)] for i in range(self.step)] for k in range(6)]
        self.EES = [[LinearNDInterpolator([point[0] for point in self.FDataE[k][i]], [point[1] for point in self.FDataE[k][i]]) for i in range(self.step)] for k in range(6)]


        self.FDataP = [[[[xyzDP[i][j][0], xyzDP[i][j][1][k]] for j in range(self.nnm)] for i in range(self.step)] for k in range(6)]
        self.PES = [[LinearNDInterpolator([point[0] for point in self.FDataP[k][i]], [point[1] for point in self.FDataP[k][i]]) for i in range(self.step)] for k in range(6)]

        self.FDataA = [[[[xyzDA[i][j][0], xyzDA[i][j][1][k]] for j in range(self.nnm)] for i in range(self.step)] for k in range(6)]
        self.AES = [[LinearNDInterpolator([point[0] for point in self.FDataA[k][i]], [point[1] for point in self.FDataA[k][i]]) for i in range(self.step)] for k in range(6)]

        step_values = np.arange(0,self.step)
        self.f_sigma = interpolate.interp1d(self.sigma_nom_list, step_values, kind='linear')

        # Define the variables you want to save
        data_to_save = {
            'EE': self.E,
            'ν': self.ν,
            'AA': self.AA,
            'nnm': self.nnm,
            'step': self.step,
            'sigma_nom_list': self.sigma_nom_list,
            'f_sigma': self.f_sigma,
            # 'DataEES': self.DataEES,
            # 'DataPES': self.DataPES,
            # 'DataAES': self.DataAES,
            # 'FDataE': self.FDataE,
            # 'FDataP': self.FDataP,
            # 'FDataA': self.FDataA, 
            'EES': self.EES,
            'PES': self.PES,
            'AES': self.AES    
            }

        # Specify the file name for saving 
        pkl_file_name =self.BasicParams.steel + "_"+ self.BasicParams.type + "_FieldValues.pkl"

        # Save the data to a file
        with open(pkl_file_name, 'wb') as file:
            pickle.dump(data_to_save, file)

        return print("Complete Definition of FieldValues Function :", datetime.datetime.now())
#Part2 End
#
#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------3. Material data import --------------------------------------------------------------            
# -------------------------------------------------------------------------------------------------------------------------------------------            

class MaterialDataImporter:
    def __init__(self, BasicParams):
        self.BasicParams = BasicParams


        # Import ferrite grain size from csv file

        self.FirstGrainDF = pd.read_csv(f"{self.BasicParams.steel}_{BasicParams.FirstGrain} grain size.csv", header=None).to_numpy()
        self.FirstGrainDFID = np.unique(self.FirstGrainDF[:, 0])
        self.FirstGrainMax = self.FirstGrainDF[-1]
        self.FirstGrainAspectDF = pd.read_csv(f"{self.BasicParams.steel}_{BasicParams.FirstGrain} grain aspect ratio.csv", header=None).to_numpy()
        angle_file = os.path.join(f"{self.BasicParams.steel}_{self.BasicParams.FirstGrain} grain angle.csv")
        if os.path.exists(angle_file):
            self.FirstGrainAngleDF = pd.read_csv(angle_file, header=None).to_numpy()
            
        # Import perlite grain size from csv file
        if BasicParams.SecondGrain == "Pearlite":
            self.Pearlite_thickness_files = [file for file in os.listdir() if file.startswith(f"{self.BasicParams.steel}_{BasicParams.SecondGrain} thickness") and file.endswith(".csv")]
            self.SecondGrainDF = pd.read_csv(self.Pearlite_thickness_files[0], header=None).to_numpy()
            #distribution fraction
            self.SecondGrainCDF = self.makeCDF(self.SecondGrainDF)
            self.SecondGrainCDFr = self.makeCDFr(self.SecondGrainDF)
            #Pearlite fraction
            # SecondGrain fraction
            self.PRate = None
            self.Pearlite_fraction()

        # distribution fraction
        self.FirstGrainCDF = self.makeCDF(self.FirstGrainDF)
        self.FirstGrainCDFr = self.makeCDFr(self.FirstGrainDF)
        self.FirstGrainAspectCDFr = self.makeCDFrA(self.FirstGrainAspectDF)
        self.FirstGrainAngleCDF = self.makeCDF(self.FirstGrainAngleDF)

        # FirstGrain average and maximum grain size
        self.dave = sum(item[0] ** 3 * item[1] for item in self.FirstGrainDF) / sum(item[0] ** 2 * item[1] for item in self.FirstGrainDF)
        self.dmax = self.FirstGrainCDF(1)
        self.ngAe = (4 * self.BasicParams.active_element_area) / (np.pi * self.dave**2)
        print("dave", self.dave)
        print("dmax", self.dmax)


        #size of the model
        self.Ng = None
        self.Mg = None



        # Monotonic tensile properties
        self.σ_YS = None
        self.σ_UT = None
        self.σ_0 = None
        self.Reduction_in_Area = None

        # Friction strength of each phase
        self.σ_fF = None
        self.σ_fP = None

        # Abaqus data
        self.E = None
        self.ν = None
        self.AA = None
        self.nnm = None
        self.step = None
        self.sigma_nom_list = None
        self.f_sigma = None
        self.DataEES = None
        self.DataPES = None
        self.DataAES = None
        self.FDataE = None
        self.FDataP = None
        self.FDataA = None
        self.EES = None
        self.PES = None
        self.AES = None

        #Constants used in the weight function
        self.sec_ell = None
        self.WFClist = None
        self.WFrbf_low = None #INTERPOLATOR
        self.WFrbf_high = None ##INTERPOLATOR
        self.values = None
        self.gauss10 = np.array([0.06667134430869041, 0.14945134915057556, 0.21908636251598135, 0.2692667193099942, 0.2955242247147529, 0.2955242247147529, 0.26926671930999607, 0.21908636251598135, 0.1494513491505782, 0.06667134430869041])
        self.gp10 = np.array([-0.9739065285171692, -0.8650633666889868, -0.679409568299024, -0.4333953941292474, -0.14887433898163122, 0.14887433898163122, 0.43339539412924727, 0.679409568299024, 0.8650633666889848, 0.9739065285171692])
        #Remote stress
        self.CFlist = None

        #
        self.gData = None
        self.pData = None
        self.PRateN = None

        #call fuctions
        self.monotonic_tensile_properties()
        self.friction_strength()
        self.abaqus_data_read()
        self.weight_function_constants()
        self.weight_function()
        self.remote_stress_constants()
        self.CreateGrainData()
        self.ModelSize()

    #CDF function
    def makeCDF(self, DF): #  Definition of the function makeCDF to calculate the cumulative distribution function of particle size
        CDF0 = np.cumsum(DF[:, 1])
        CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
        return interp1d(CDF, DF[:, 0], kind='linear', fill_value='extrapolate')

    def makeCDFr(self, DF): #  Definition of the function makeCDF to calculate the cumulative distribution function of particle size
        CDF0 = np.cumsum(DF[:, 1])
        CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
        return interp1d(CDF, np.sqrt(np.pi) / 2. * DF[:, 0], kind='linear', fill_value='extrapolate')

    def makeCDFrA(self, DF):
        CDF0 = np.cumsum(DF[:, 1])
        CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
        RA = np.sqrt(DF[:, 0])
        return interp1d(CDF, RA, kind='linear', fill_value='extrapolate')
    
    #Volume fraction of Pearlite
    def Pearlite_fraction(self):
        pattern = re.compile(r"(\d+\.\d+)") 
        match = re.search(pattern, str(self.Pearlite_thickness_files))
        if match:
            numeric_part = match.group(1)
            CRate = float(numeric_part)
        else:
            print("No numeric part found in the file name.")
        self.PRate = 1.5 * CRate

    #Monotonic tensile properties
    def monotonic_tensile_properties(self):
        tensile_test_data = pd.read_csv(self.BasicParams.steel + "_Monotonic tensile test.csv")
        self.σ_YS, self.σ_UT, self.Reduction_in_Area = tensile_test_data.iloc[0, [0, 1, 2]]
        self.σ_0 = 0.5 * (self.σ_YS + self.σ_UT)

    #Friction strength of each phase
    def friction_strength(self):
        friction_strength_data = pd.read_csv(self.BasicParams.steel + "_Friction strength.csv")
        if self.BasicParams.steel in ["H"]:
            self.σ_fF = friction_strength_data.iloc[0, 0]
        else:
            self.σ_fF, self.σ_fP = friction_strength_data.iloc[0, [0, 1]]

    #Abaqus data
    def abaqus_data_read(self):
        pkl_file_name =self.BasicParams.steel + "_"+ self.BasicParams.type + "_FieldValues.pkl"
        with open(pkl_file_name, 'rb') as file:
            data = pickle.load(file)

        variable_names = list(data.keys())
        print("  Stored variable names:", variable_names)

        # Access to specific variables    
        self.EE = data['EE']
        self.ν = data['ν']
        self.AA = data['AA']
        self.nnm = data['nnm']
        self.step = data['step']
        self.sigma_nom_list = data['sigma_nom_list']
        self.f_sigma = data['f_sigma']
        # self.DataEES = data['DataEES']
        # self.DataPES = data['DataPES']
        # self.DataAES = data['DataAES']
        # self.FDataE = data['FDataE']
        # self.FDataP = data['FDataP']
        # self.FDataA = data['FDataA']
        self.EES = data['EES']
        self.PES = data['PES']
        self.AES = data['AES']
        print("Complete Reading ABAQUS FieldValues Function from pkl File :", datetime.datetime.now())
    

    #Weight function

    def weight_function_constants(self):
        with open("CombinedData_Ellipse_WF_CF.json") as json_file1:
            Ellipse_WF_CF = json.load(json_file1)

        ldh = Ellipse_WF_CF["ldh"]
        WFCsmooth = Ellipse_WF_CF["WFCsmooth"]
        WFC3PBT = Ellipse_WF_CF["WFC3PBT"]
        WFCCS = Ellipse_WF_CF["WFCCS"]
        # Creating Interpolation Functions
        x_values = [item[0] for item in ldh]
        y_values = [item[1] for item in ldh]
        self.sec_ell = interpolate.interp1d(x_values, y_values)

        # Calculation of K-values via weight functions
        if self.BasicParams.type in ["Smooth"]:
            self.WFClist = WFCsmooth
        elif self.BasicParams.type in ["R008", "R100"]:
            self.WFClist = WFC3PBT
        elif self.BasicParams.type  in ["CS","CSS"]:
            self.WFClist = WFCCS

    def weight_function(self):
        at_ac_values = [item[0] for item in self.WFClist]
        values = [item[1] for item in self.WFClist]

        at_values, ac_values = zip(*at_ac_values)
        at_values = np.array(at_values)
        ac_values = np.array(ac_values)
        values = np.array(values)

        # 分区数据
        if np.max(at_ac_values)>0.8:
            mask_low = at_values <= 0.8
            mask_high = at_values >= 0.6



            if np.any(mask_low):
                self.WFrbf_low = RBFInterpolator(
                    np.column_stack((at_values[mask_low], ac_values[mask_low])),
                    values[mask_low], kernel='thin_plate_spline', epsilon=1
                )

            if np.any(mask_high):
                self.WFrbf_high = RBFInterpolator(
                    np.column_stack((at_values[mask_high], ac_values[mask_high])),
                    values[mask_high], kernel='thin_plate_spline', epsilon=1
                )
        else:
            self.WFrbf = RBFInterpolator(np.column_stack((at_values, ac_values)), values, kernel='thin_plate_spline', epsilon=1)

    
    
    def QQ(self,n):
        if n <= 1:
            return 1 + 1.464 * n ** 1.65
        else:
            return n ** 2 * (1 + 1.464 * (1 / n) ** 1.65)
            
    def remote_stress_constants(self):
        with open("CombinedData_Ellipse_WF_CF.json") as json_file1:
            Ellipse_WF_CF = json.load(json_file1)
        # Extracting data for different variables
        CFsmooth = Ellipse_WF_CF["CFsmooth"]
        CF3PBT = Ellipse_WF_CF["CF3PBT"]
        CFCS = Ellipse_WF_CF["CFCS"]
        if self.BasicParams.type in ["Smooth"]:
            self.CFlist = CFsmooth
        elif BasicParams.type in ["R008", "R100"]:
            self.CFlist = CF3PBT    
        elif BasicParams.type in ["CS","CSS"]:
            self.CFlist = CFCS


        # Calculate CFlist2
        points = [(row[0], row[1]) for row in self.CFlist]
        values = [(row[2] * np.sqrt(self.QQ(row[1]))) / np.sqrt(row[0] * self.BasicParams.thickness) for row in self.CFlist]
        self.σ_CF = lambda x, y: griddata(points, values, (x, y), method='cubic')

    

        




    def CreateGrainData(self):
        def VolumeFraction():#
            if self.BasicParams.steel in [ "Bainite", "Martensite",'H']:
                self.PRateN = 0
            else:
                # Generating Random Numbers
                VP = np.mean([self.SecondGrainCDFr(r) * self.FirstGrainCDFr(r) for r in [random.random() for _ in range(100000)]])

                VF = np.mean([self.FirstGrainCDFr(random.random()) ** 2 for _ in range(100000)])
                self.PRateN = (self.PRate / VP) / (self.PRate / VP + (1 - self.PRate) / VF)   
                print("VP:", VP)
                print("VF:", VF)
                print("PRateN:", self.PRateN)  
        VolumeFraction()


        def makegList(r, d, t, ra, ang):
            if r > self.PRateN:
                if self.BasicParams.type == "CS":
                    return [
                        self.σ_fF,
                        d * np.sqrt(np.sqrt((np.cos(ang)**2 + ra**4 * np.sin(ang)**2) / (np.sin(ang)**2 + ra**4 * np.cos(ang)**2))),
                        d * np.sqrt(np.sqrt((np.sin(ang)**2 + ra**4 * np.cos(ang)**2) / (np.cos(ang)**2 + ra**4 * np.sin(ang)**2)))
                    ]
                else:
                    return [
                        self.σ_fF,
                        d * np.sqrt(np.sqrt((np.sin(ang)**2 + ra**4 * np.cos(ang)**2) / (np.cos(ang)**2 + ra**4 * np.sin(ang)**2))),
                        d * np.sqrt(np.sqrt((np.cos(ang)**2 + ra**4 * np.sin(ang)**2) / (np.sin(ang)**2 + ra**4 * np.cos(ang)**2)))
                    ]
            else:
                if self.BasicParams.type == "CS":
                    return [
                        self.σ_fP,
                        math.sqrt(t * d * ra * math.sqrt((t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2) /(t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2))),
                        math.sqrt(t * d * ra * math.sqrt((t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2) /(t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2)))
                    ]
                else:
                    return [
                        self.σ_fP,
                        math.sqrt(t * d * ra * math.sqrt((t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2) /(t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2))),
                        math.sqrt(t * d * ra * math.sqrt((t**2 * math.cos(ang)**2 + d**2 * ra**2 * math.sin(ang)**2) /(t**2 * math.sin(ang)**2 + d**2 * ra**2 * math.cos(ang)**2))),

                    ]
        

        if os.path.exists(self.BasicParams.steel + "_"+ self.BasicParams.type + "_gData.pkl"):
            with open(self.BasicParams.steel + "_"+ self.BasicParams.type + "_gData.pkl", "rb") as file:
                self.gData = pickle.load(file)
                print("Finish loading gData.pkl", datetime.datetime.now())
        else:
            print("Start generating gData.pkl", datetime.datetime.now())
            nData = 1000000
            R = np.random.rand(5, nData)
            if self.BasicParams.steel in ["Bainite", "Martensite", "H"]:
                dtrList = [[row[0], self.FirstGrainCDFr(row[2]),np.array(0), self.FirstGrainAspectCDFr(row[1])] for row in R.T]
            else: 
                if self.BasicParams.type == "CS":
                    dtrList = [[row[0], self.FirstGrainCDFr(row[1]), self.SecondGrainCDFr(row[2]), self.FirstGrainAspectCDFr(row[3]), np.pi / 180 * self.FirstGrainAngleCDF(row[4])] for row in R.T]
                else:
                    dtrList = [[row[0], self.FirstGrainCDFr(row[1]), self.SecondGrainCDFr(row[1]), self.FirstGrainAspectCDFr(row[3]), np.pi / 180 * self.FirstGrainAngleCDF(row[4])-np.pi/2] for row in R.T]

            self.gData = list(map(lambda x: makegList(*x), dtrList))#Generate a list of nData grains (containing ferite and SecondGrain)
            print("Finish generating gData.pkl", datetime.datetime.now())

            with open(self.BasicParams.steel + "_"+ self.BasicParams.type + "_gData.pkl", "wb") as file:
                pickle.dump((self.gData), file)
        gData_subset = self.gData[:2000]

        df = pd.DataFrame(gData_subset)

        csv_filename = self.BasicParams.steel + "_" + self.BasicParams.type + "_gData_2000.csv"
        df.to_csv(csv_filename, index=False)

        print("Finish saving first 2000 rows of gData to CSV", datetime.datetime.now())
###for ES   

        if BasicParams.type in ["CS","CSS"]:
            if os.path.exists(self.BasicParams.steel + "_"+ self.BasicParams.type + "_pData.pkl"):
                with open(self.BasicParams.steel + "_"+ self.BasicParams.type + "_pData.pkl", "rb") as file1:
                    self.pData = pickle.load(file1)
                    print("Finish loading pData.pkl", datetime.datetime.now())

            else:
                print("Start generating PData.pkl", datetime.datetime.now())
                nData = 1000000
                pR = np.random.uniform(self.PRateN - 1/1000, self.PRateN, (1, nData))
                lR = np.random.uniform(0, 1, (3, nData))
                R = np.vstack((pR, lR))
                
                dtrpList = []
                for i in range(nData):
                    ferrite_angle = np.pi / 180 * self.FirstGrainAngleCDF(R[3, i])
                    dtrpList.append([
                        R[0, i],
                        self.FirstGrainCDFr(R[1, i]),
                        self.SecondGrainCDFr(R[2, i]),
                        self.FirstGrainAspectCDFr(R[3, i]),
                        ferrite_angle
                    ])
                self.pData = [makegList(*d) for d in dtrpList]    
                print("Finish generating PData.pkl", datetime.datetime.now())
                with open(self.BasicParams.steel + "_"+ self.BasicParams.type + "_pData.pkl", "wb") as file:
                    pickle.dump((self.pData), file)   





    def ModelSize(self):#   Model size
        # Generate the first part of A0
        part1 = [(self.FirstGrainCDF(random.random()))**2 for i in range(int(50000 * (1 - self.PRateN)))]
        # Generate the second part of A0
        part2 = [self.FirstGrainCDF(random.random()) * self.SecondGrainCDF(random.random()) for i in range(int(50000 * self.PRateN))]
        # Combine the two parts to create A0
        A0 = part1 + part2

        fw = 6 if self.BasicParams.steel == "Smooth" else 10
        self.Ng = round(fw * ((2 * self.BasicParams.thickness**2) /np.mean(A0))) # Ng is the sufficient large number of grains in the area of fracture surface

        d0 = np.mean([self.FirstGrainCDF(random.random()) * self.FirstGrainAspectCDFr(random.random()) for i in range(50000)])
        self.Mg = round((self.BasicParams.thickness * fw) / d0) # Mg is the sufficient large number of layers in the thickness direction
#Part3 End
#
#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------4. Creat Field value function ------------------------------------------------------           
# -------------------------------------------------------------------------------------------------------------------------------------------    

class FieldValuesFunction:
    def __init__(self, BasicParams, DataImport,load_queue):
        self.BasicParams = BasicParams
        self.DataImport = DataImport
        self.x_Present = None
        self.x_steps_number = None
        self.stress_active_ele = None
        self.FieldValues_ACTIVE = None
        self.AeYZ = None
        self.load_queue = load_queue


        #call fuctions
        self.x_Present_creator()
        self.FData_AE()
        self.FData_AEE()
        self.FData_AEP()
# -------------------------------------------------------------------------------------------------------------------------------------------
    def FieldValuesA(self,x, y, z):
        y = np.array(y)
        z = np.array(z)
        s = self.DataImport.f_sigma(self.BasicParams.σ_nom[self.load_queue])
        s1 = int(np.floor(s))
        s2 = int(np.ceil(s))
        if BasicParams.type == "SmoothT": #symmetrical in thickness
            mask = x <= self.BasicParams.thickness / 2
            x1, y1, z1 = x[mask], y[mask], z[mask]
            x2, y2, z2 = x[~mask], y[~mask], z[~mask]

            #计算x1的应力
            if s - s1 <= 0.01:
                result_ori = [self.DataImport.AES[k][s1](x1, y1, z1) for k in range(6)]
                result_mirrored = [self.DataImport.AES[k][s1](self.BasicParams.thickness - x2, y2, z2) for k in range(6)]

            elif s2 - s <= 0.01:
                result_ori = [self.DataImport.AES[k][s2](x1, y1, z1) for k in range(6)]
                result_mirrored = [self.DataImport.AES[k][s2](self.BasicParams.thickness - x2, y2, z2) for k in range(6)]

            else:
                result_ori = [(s2 - s) * self.DataImport.AES[k][s1](x1, y1, z1) + (s - s1) * self.DataImport.AES[k][s2](x1, y1, z1) for k in range(6)]
                result_mirrored = [(s2 - s) * self.DataImport.AES[k][s1](self.BasicParams.thickness - x2, y2, z2) + (s - s1) * self.DataImport.AES[k][s2](self.BasicParams.thickness - x2, y2, z2) for k in range(6)]

        # Adjust mirrored results for symmetry
            result_mirrored[3]= -result_mirrored[3]
            result_mirrored[4]= -result_mirrored[4]
            result = np.hstack((result_ori, result_mirrored))
        else:
            if s - s1 <= 0.01:
                result = [self.DataImport.AES[k][s1](x, y, z) for k in range(6)]
            elif s2 - s <= 0.01:
                result = [self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
            else:
                result = [(s2 - s) * self.DataImport.AES[k][s1](x, y, z) + (s - s1) * self.DataImport.AES[k][s2](x, y, z) for k in range(6)]

        return result
    
#elastic or plastic only
    def FieldValuesAE(self,x, y, z):
        y = np.array(y)
        z = np.array(z)
        s = self.DataImport.f_sigma(self.BasicParams.σ_nom[self.load_queue])
        s1 = int(np.floor(s))
        s2 = int(np.ceil(s))

        if s - s1 <= 0.01:
            result = [self.DataImport.EES[k][s1](x, y, z) for k in range(6)]
        elif s2 - s <= 0.01:
            result = [self.DataImport.EES[k][s2](x, y, z) for k in range(6)]
        else:
            result = [(s2 - s) * self.DataImport.EES[k][s1](x, y, z) + (s - s1) * self.DataImport.EES[k][s2](x, y, z) for k in range(6)]
        return result
    
    def FieldValuesAP(self,x, y, z):
        y = np.array(y)
        z = np.array(z)
        s = self.DataImport.f_sigma(self.BasicParams.σ_nom[self.load_queue])
        s1 = int(np.floor(s))
        s2 = int(np.ceil(s))

        if s - s1 <= 0.01:
            result = [self.DataImport.PES[k][s1](x, y, z) for k in range(6)]
        elif s2 - s <= 0.01:
            result = [self.DataImport.PES[k][s2](x, y, z) for k in range(6)]
        else:
            result = [(s2 - s) * self.DataImport.PES[k][s1](x, y, z) + (s - s1) * self.DataImport.PES[k][s2](x, y, z) for k in range(6)]
        return result
    


    def FieldValuesAOpen(self,σ, x, y, z):
        s = self.DataImport.f_sigma(σ)
        s1 = int(np.floor(s))
        s2 = int(np.ceil(s))

        if BasicParams.type == "SmoothT": #symmetrical in thickness
            if x <= self.BasicParams.thickness / 2:
                if s - s1 <= 0.01:
                    result = [self.DataImport.AES[k][s1](x, y, z) for k in range(6)]
                elif s2 - s <= 0.01:
                    result = [self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
                else:
                    result = [(s2 - s) * self.DataImport.AES[k][s1](x, y, z) + (s - s1) * self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
            else: #get result in symmetrical positions
                if s - s1 <= 0.01:
                    mirrored_result = [self.DataImport.AES[k][s1](self.BasicParams.thickness - x, y, z) for k in range(6)]
                elif s2 - s <= 0.01:
                    mirrored_result = [self.DataImport.AES[k][s2](self.BasicParams.thickness - x, y, z) for k in range(6)]
                else:
                    mirrored_result = [(s2 - s) * self.DataImport.AES[k][s1](self.BasicParams.thickness - x, y, z) + (s - s1) * self.DataImport.AES[k][s2](self.BasicParams.thickness - x, y, z) for k in range(6)]
                result = mirrored_result.copy()
                result[3] = -mirrored_result[3]  # σxy
                result[4] = -mirrored_result[4]  # σxz
        else:
            if s - s1 <= 0.01:
                result = [self.DataImport.AES[k][s1](x, y, z) for k in range(6)]
            elif s2 - s <= 0.01:
                result = [self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
            else:
                result = [(s2 - s) * self.DataImport.AES[k][s1](x, y, z) + (s - s1) * self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
        return result


    #generate x_Present for 1D interpolation
    def x_Present_creator(self):
        precision=2
        start = 0
        end = BasicParams.thickness#modify

        # Define the initial step size and growth factor
        initial_step = 0.01
        step_growth_factor = 1

        # Initialize an empty list to store data
        self.x_Present = []

        # Gradually increase the step size and generate data
        current_step = initial_step
        current_value = start

        while current_value < end:
            self.x_Present.append(round(current_value, precision))
            current_value += current_step
            current_step *= step_growth_factor

        self.x_Present = np.array(self.x_Present)
        self.x_Present_number=len(self.x_Present)


    # Definition of the function FDataPAE to calculate the stress and strain of each area element
    def FData_AE(self):

        AeNumY = int(self.BasicParams.y_lim / self.BasicParams.active_element_size[0])
        AeNumZ = int(self.BasicParams.z_lim / self.BasicParams.active_element_size[1])

        AeNum = AeNumY * AeNumZ  # Number of area elements in the target area

        # Calculate the centre coordinates of each area element
        self.AeYZ = np.array([[self.BasicParams.active_element_size[0] * (i - 0.5 if i > 0 else 0), 
                               self.BasicParams.active_element_size[1] * (j - 0.5 if j > 0 else 0)]
                      for i in range(AeNumY + 1)  # Start from 0 to include points with y-coordinate as 0
                      for j in range(AeNumZ + 1)])

        y_Present = [[self.AeYZ[i, 0] for _ in range(self.x_Present_number)] for i in range(len(self.AeYZ))]
        z_Present = [[self.AeYZ[i, 1] for _ in range(self.x_Present_number)] for i in range(len(self.AeYZ))]
        stress_active0 = [self.FieldValuesA(self.x_Present, y_Present[i], z_Present[i])for i in range(len(self.AeYZ))]
        self.stress_active_ele=np.array(stress_active0)
        
    def FData_AEE(self):#ELASTIC

        AeNumY = int(self.BasicParams.y_lim / self.BasicParams.active_element_size[0])
        AeNumZ = int(self.BasicParams.z_lim / self.BasicParams.active_element_size[1])

        AeNum = AeNumY * AeNumZ  # Number of area elements in the target area

        # Calculate the centre coordinates of each area element
        self.AeYZ = np.array([[self.BasicParams.active_element_size[0] * (i - 0.5 if i > 0 else 0), 
                               self.BasicParams.active_element_size[1] * (j - 0.5 if j > 0 else 0)]
                      for i in range(AeNumY + 1)  # Start from 0 to include points with y-coordinate as 0
                      for j in range(AeNumZ + 1)])

        y_Present = [[self.AeYZ[i, 0] for _ in range(self.x_Present_number)] for i in range(len(self.AeYZ))]
        z_Present = [[self.AeYZ[i, 1] for _ in range(self.x_Present_number)] for i in range(len(self.AeYZ))]
        stress_active0 = [self.FieldValuesAE(self.x_Present, y_Present[i], z_Present[i])for i in range(len(self.AeYZ))]
        self.stress_active_ele_E=np.array(stress_active0)

    def FData_AEP(self):

        AeNumY = int(self.BasicParams.y_lim / self.BasicParams.active_element_size[0])
        AeNumZ = int(self.BasicParams.z_lim / self.BasicParams.active_element_size[1])

        AeNum = AeNumY * AeNumZ  # Number of area elements in the target area

        # Calculate the centre coordinates of each area element
        self.AeYZ = np.array([[self.BasicParams.active_element_size[0] * (i - 0.5 if i > 0 else 0), 
                               self.BasicParams.active_element_size[1] * (j - 0.5 if j > 0 else 0)]
                      for i in range(AeNumY + 1)  # Start from 0 to include points with y-coordinate as 0
                      for j in range(AeNumZ + 1)])

        y_Present = [[self.AeYZ[i, 0] for _ in range(self.x_Present_number)] for i in range(len(self.AeYZ))]
        z_Present = [[self.AeYZ[i, 1] for _ in range(self.x_Present_number)] for i in range(len(self.AeYZ))]
        stress_active0 = [self.FieldValuesAP(self.x_Present, y_Present[i], z_Present[i])for i in range(len(self.AeYZ))]
        self.stress_active_ele_P=np.array(stress_active0)

    def FieldValues_ACTIVE_Numpy(self,x, y, z):
        for i in range(len(self.AeYZ)):
            if self.AeYZ[i, 0] == y and self.AeYZ[i, 1] == z:
                location = i
                break

        return [np.interp(x, self.x_Present,self.stress_active_ele[location][k])for k in range(6)]
    def FieldValues_ACTIVE_NumpyE(self,x, y, z):
        for i in range(len(self.AeYZ)):
            if self.AeYZ[i, 0] == y and self.AeYZ[i, 1] == z:
                location = i
                break

        return [np.interp(x, self.x_Present,self.stress_active_ele_E[location][k])for k in range(6)]
    def FieldValues_ACTIVE_NumpyP(self,x, y, z):
        for i in range(len(self.AeYZ)):
            if self.AeYZ[i, 0] == y and self.AeYZ[i, 1] == z:
                location = i
                break

        return [np.interp(x, self.x_Present,self.stress_active_ele_P[location][k])for k in range(6)]
    

#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------5. Life evaluation for single crack------------------------------------------------------           
# -------------------------------------------------------------------------------------------------------------------------------------------    

def CrackLifeCalc(r0, or1, y, z, Anai, FieldValues,S_cyc_result, S_ai_result):

                    

    nTemp = S_cyc_result[-1]
    Scyc1st = list(zip(S_ai_result, S_cyc_result)) + [[BasicParams.thickness, S_cyc_result[-1]]]
    LifeMin = lambda x: np.interp(x, [row[0] for row in Scyc1st], [row[1] for row in Scyc1st], left=1e15, right=1e15)

#   get r1,rt1
    def get_r1_rt1(): 
        aspg = DataImport.FirstGrainAspectCDFr(random.random())
        r1 = aspg * r0*0.5
        rt1 = r0*0.5/aspg
        return r1, rt1
    r1, rt1 = get_r1_rt1()


    # get crack aspect
    def getCrackAspectFunc(r1, rt1, Anai):
            # Definition of a function that returns the crack aspect ratio from the crack depth at specific crack length
            # r1:crack depth，rt1：crack length，Anai：crack aspect ratio
        def F2(n):
            return 1 + 0.7637 * n**2 + 0.2604 * n**4 + 0.04296 * n**6 + 0.002796 * n**8

        def Wu(a, a0, rr0):
            if a == 0:
                return 1
            elif a0 == 0:
                return F2(a)**(-1/3)
            else:
                return (F2(a) - (a0 / a)**3 * (F2(a0) - rr0**(-3)))**(-1/3)

        def ff000(a):
            wu1 = Wu(a, r1 / BasicParams.thickness, r1 / rt1)
            wu2 = Wu(a, 0, 0)
            wu3 = Wu(r1, r1 / BasicParams.thickness, r1 / rt1)
            wu4 = Wu(r1, 0, 0)
            return ((wu1 - wu2) / (wu3 - wu4)) * (wu3 - Anai(r1 / BasicParams.thickness)) + Anai(a / BasicParams.thickness)

        def ff00(a):
            if a < 0.9 * BasicParams.thickness:
                return 1 / ff000(a)
            else:
                return 1 / ff000(0.9 * BasicParams.thickness)
        return ff00
    



    def generateGrains(r1, rt1, ff00):#
#----------------------------------------------sub_function__START------------------------------------------------
        def L2(asp0, b): # Definition of a function that returns the arc length from the aspect ratio and crack depth
            asp1 = float(asp0)

            if b * asp0 < BasicParams.width:
                if asp0 >= 1:
                    # print("b * asp1 * DataImport.sec_ell(asp1):", b * asp1 * DataImport.sec_ell(asp1))
                    return b * asp1 * DataImport.sec_ell(asp1)
                else:
                    # print("b * DataImport.sec_ell(1/asp1):", b * DataImport.sec_ell(1/asp1))

                    return b * DataImport.sec_ell(1/asp1)
            else:
                # print("BasicParams.width :", BasicParams.width )
                return BasicParams.width  
    
#----------------------------------------------sub_function__END-----------------------------------------------
        if BasicParams.type in ["CS","CSS"]:
            g_List = np.array([[DataImport.σ_fF, r1, rt1]] + [random.choice(DataImport.gData) for _ in range(DataImport.Ng)])
            p_List = np.array([random.choice(DataImport.pData) for _ in range(DataImport.Ng)])#+++

            σ_f_List, dnList, tnList = g_List[:, 0], g_List[:, 1], g_List[:, 2]
            σ_p_List, dnpList, tnpList = p_List[:, 0], p_List[:, 1], p_List[:, 2]#+++

            orpList = [or1] + [BasicParams.makeEulerAngles() for _ in range(DataImport.Ng)]# 
        else:
            g_List = np.array([[DataImport.σ_fF, r1, rt1]] + [random.choice(DataImport.gData) for _ in range(DataImport.Ng)])

        σ_f_List, dnList, tnList = g_List[:, 0], g_List[:, 1], g_List[:, 2]
        orList = [or1] + [BasicParams.makeEulerAngles() for _ in range(DataImport.Ng)]# 

        rnList = [0] * DataImport.Mg  
        Nd = [0] *DataImport. Mg   
        RnA = [[] for _ in range(DataImport.Mg)] 
        RnAA = [[] for _ in range(DataImport.Mg)]  
        PnList = [[] for _ in range(DataImport.Mg)]
        RnAS = [[] for _ in range(DataImport.Mg)]   
        RnAAS = [[] for _ in range(DataImport.Mg)]  

        rnList[0] = r1
        Nd[0] = 1
        RnA[0] = [1]
        RnAA[0] = [1]

        n = 0
        n0 = 0 #First grain number containing the crack tip
        n1 = 0 #Last grain number containing the crack tip

        ff00 = getCrackAspectFunc(r1, rt1, Anai)

        if BasicParams.type == "CS":#+++
            pCol=[]
        else:
            pCol=[]

        while rnList[n] < 1.05 * BasicParams.thickness:

            rr = rnList[n] 
            m0 = n1 + 1
            if m0 >= DataImport.Ng:
                break
            m02 = m0 + (n1 - n0)
            if m02 >= DataImport.Ng:
                m02 = DataImport.Ng
            gr = 0
            nnn = 1

            if (n + 1) in pCol:

                if rr < BasicParams.thickness * 0.9:
                
                    rr0 = rr + np.dot(dnList[m0 : m02+1], tnList[m0 : m02+1]) / (2 * np.sum(tnList[m0 : m02+1]))

                    if L2(ff00(rr0), rr0) < np.sum(tnList[m0:m02+1]):
                        nnn = 1
                        while True:
                            if (m02 - nnn) >= m0:
                                dn_slice = dnList[m0:m02 - nnn + 1]
                                tn_slice = tnList[m0:m02 - nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(rr_temp), rr_temp) > np.sum(tn_slice):
                                    break
                            else:
                                break
                            
                            nnn += 1
                        gr = 1

                    else:
                        nnn = 1

                        while True:
                            if (m02 + nnn) <= DataImport.Ng:
                                dn_slice = dnList[m0:m02 + nnn + 1]
                                tn_slice = tnList[m0:m02 + nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(rr_temp), rr_temp) < np.sum(tn_slice):
                                    break
                            else:
                                break
                            nnn += 1
                        gr = 2
                    
                else:
                    rr0 = rr + np.dot(dnList[m0:m02+1], tnList[m0:m02+1]) / (2 * np.sum(tnList[m0:m02+1]))

                    if L2(ff00(BasicParams.thickness * 0.9), rr0) < np.sum(tnList[m0:m02+1]):
                        nnn = 1
                        while True:
                            if (m02 - nnn) >= m0:
                                dn_slice = dnList[m0:m02 - nnn + 1]
                                tn_slice = tnList[m0:m02 - nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(BasicParams.thickness * 0.9), rr_temp) > np.sum(tn_slice):
                                    break
                            else:
                                break                            
                            nnn += 1
                        gr = 1
                    else:
                        nnn = 1
                        while True:
                            if (m02 + nnn) <= DataImport.Ng:
                                dn_slice = dnList[m0:m02 + nnn + 1]
                                tn_slice = tnList[m0:m02 + nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(BasicParams.thickness * 0.9), rr_temp) < np.sum(tn_slice):
                                    break
                            else:
                                break
                            nnn += 1
                        gr = 2
                # Compute pn0 and pn1
                pn0 = n1 + 1  # The first grain number containing the crack tip

                if gr == 1:
                    pn1 = m02 - nnn + 1
                else:
                    pn1 = m02 + nnn - 1

                # Update dnList, tnList, and σ_fList from pn0 to pn1
                dnList[pn0:pn1 + 1] = dnpList[pn0:pn1 + 1]
                tnList[pn0:pn1 + 1] = tnpList[pn0:pn1 + 1]
                σ_f_List[pn0:pn1 + 1] = σ_p_List[pn0:pn1 + 1]

            else:
                if rr < BasicParams.thickness * 0.9:
                
                    rr0 = rr + np.dot(dnList[m0 : m02+1], tnList[m0 : m02+1]) / (2 * np.sum(tnList[m0 : m02+1]))

                    if L2(ff00(rr0), rr0) < np.sum(tnList[m0:m02+1]):
                        nnn = 1
                        while True:
                            if (m02 - nnn) >= m0:
                                dn_slice = dnList[m0:m02 - nnn + 1]
                                tn_slice = tnList[m0:m02 - nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(rr_temp), rr_temp) > np.sum(tn_slice):
                                    break
                            else:
                                break
                            
                            nnn += 1
                        gr = 1

                    else:
                        nnn = 1

                        while True:
                            if (m02 + nnn) <= DataImport.Ng:
                                dn_slice = dnList[m0:m02 + nnn + 1]
                                tn_slice = tnList[m0:m02 + nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(rr_temp), rr_temp) < np.sum(tn_slice):
                                    break
                            else:
                                break
                            nnn += 1
                        gr = 2
                    
                else:
                    rr0 = rr + np.dot(dnList[m0:m02+1], tnList[m0:m02+1]) / (2 * np.sum(tnList[m0:m02+1]))

                    if L2(ff00(BasicParams.thickness * 0.9), rr0) < np.sum(tnList[m0:m02+1]):
                        nnn = 1
                        while True:
                            if (m02 - nnn) >= m0:
                                dn_slice = dnList[m0:m02 - nnn + 1]
                                tn_slice = tnList[m0:m02 - nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(BasicParams.thickness * 0.9), rr_temp) > np.sum(tn_slice):
                                    break
                            else:
                                break                            
                            nnn += 1
                        gr = 1
                    else:
                        nnn = 1
                        while True:
                            if (m02 + nnn) <= DataImport.Ng:
                                dn_slice = dnList[m0:m02 + nnn + 1]
                                tn_slice = tnList[m0:m02 + nnn + 1]
                                rr_temp = rr + np.dot(dn_slice, tn_slice) / (2 * np.sum(tn_slice))
                                
                                if L2(ff00(BasicParams.thickness * 0.9), rr_temp) < np.sum(tn_slice):
                                    break
                            else:
                                break
                            nnn += 1
                        gr = 2


            n0 = n1 + 1
            n1 = m02 - nnn + 1 if gr == 1 else m02 + nnn - 1
            # print(f"gr: {gr}", f"nnn: {nnn}")
            if n1 > DataImport.Ng:
                break

            Nd[n + 1] = n1 - n0 + 1

            RnA[n + 1] = tnList[n0:n1+1] / sum(tnList[n0:n1+1])
            RnAA[n + 1] = np.cumsum(RnA[n + 1])
            rnList[n + 1] = rnList[n] + np.dot(RnA[n + 1], dnList[n0:n1+1])
            k0, k1 = 1, 1
            Pn = [0] * (Nd[n] + Nd[n + 1] - 1)

            Pn[Nd[n] + Nd[n + 1] - 2] = [Nd[n], Nd[n + 1]]
            Pn[0] = [0, 0]

            for j in range(1, Nd[n] + Nd[n + 1] - 1):
                if RnAA[n][k0 - 1] < RnAA[n + 1][k1 - 1]:
                    k0 += 1
                else:
                    k1 += 1
                Pn[j] = [k0-1, k1-1]

            PnList[n] = Pn
            RnAAS[n] = sorted(np.concatenate((RnAA[n] , RnAA[n + 1][:-1])))
            RnAS[n] = np.array(RnAAS[n]) - np.array([0] + RnAAS[n][:-1])
            n += 1
            
            # if len(PnList[0]) <= 2:
            #     print(f"n: {n}")
            #     print(f"nnn: {nnn}")
            #     print(f"PnList: {PnList}")
            #     print(f"rnList: {rnList}")
            #     print(f"Nd: {Nd}")
        rnList = np.array(rnList[:n+1])
        Nd = np.array(Nd[:n+1])


        RnA = RnAA[:n+1]
        RnAA = RnAA[:n+1]
        RnAS = RnAS[:n]
        RnAAS = RnAAS[:n]
        PnList = PnList[:n]
    





        return σ_f_List, rnList, Nd, PnList, RnAS, orList



    def evalCTSD(aii, Δσ, σ_f_List, orList, rnList, Nd, PnList, RnAS):# Calculation of CTSD for each evaluation point
        #aii : crack depth ;
        #Δσ : stress range tensor ;
        #σ_f_List : friction strength list ;
        #orList : orientation list ;
        #rnList : grain boundary depth list ;
        #Nd : number of grains in each layer ;
        #PnList : adjacent grains of each grain in each layer ;
        #RnAS : Percentage of distance between each of the two grain boundaries by combining the two layers
        #----------------------------------------------sub_function__START------------------------------------------------
        def safe_arccos(x):
            return np.arccos(np.clip(x, -1, 1))
        
        
        def CC1(a, σ_fr, rnlist):
            def Eq_1(c, a, σ_fr):
                epsilon = 1e-10 # Avoid dividing by zero
                return np.pi / 2 - σ_fr[0] * safe_arccos(a / (c + epsilon))
            
            upper_bound = rnlist + 100
            max_upper_bound = 1e8  
            while upper_bound <= max_upper_bound:
                try:
                    c_root = brentq(Eq_1, a + 1e-10, upper_bound, args=(a, σ_fr))
                    return c_root
                except ValueError:
                    upper_bound *= 10
            return c_root


        def CCn(a, n, σ_fr, rnList):
            def Eq_n(c, a, n, σ_fr, rnList):
                epsilon = 1e-10 # Avoid dividing by zero
                return np.pi * 0.5 - σ_fr[0] * safe_arccos(a / (c + epsilon)) - sum((σ_fr[i+1] - σ_fr[i]) * safe_arccos(rnList[i] / (c + epsilon)) for i in range(n+1))
            
            upper_bound = rnList[n] + 10000
            try:
                c_root = brentq(Eq_n, a + 1e-10, upper_bound, args=(a, n, σ_fr, rnList))
                return c_root
            except ValueError:
                c_root = 10000
            return c_root
        

        def gg(x, c, ad):# 
            if x == ad:
                a = (1 - 1e-3) * ad
            else:
                a = ad

            ca = np.sqrt(c**2 - a**2)
            cx = np.sqrt(c**2 - x**2)
            return a * np.log(abs((ca + cx) / (ca - cx))) - x * np.log(abs((x * ca + a * cx) / (x * ca - a * cx)))

        def CTSD(a, c, n, σ_fr, Δτ_j, rnList_L):#calculation of CTSD

            if a == 0:
                term1 = 0
            else:
                term1 = 2 * a * σ_fr[0] * np.log(c / a)
        
            term2 =  sum((σ_fr[i + 1] - σ_fr[i]) * gg(a, c, rnList_L[i]) for i in range(n+1))
            result = Δτ_j / (np.pi ** 2 * DataImport.AA) * (term1 + term2)
            return result
        #----------------------------------------------sub_function__END------------------------------------------------

        unstable = 0
        goto=0

        rnList_temp = rnList.tolist()
        jj = rnList_temp.index(next(filter(lambda x: x > aii, rnList_temp)))+1
        N0 = sum(Nd[0 : jj])


        if jj == 1:
            τ1, θn1, θs1 = BasicParams.SlipPlane(orList[0], Δσ)

            t_List = [ [] for _ in range(DataImport.Mg) ]
            τ_List = [ [] for _ in range(DataImport.Mg) ]
            σ_fr = [0] * DataImport.Mg
            t_List[0] = [τ1 * np.outer(θn1, θs1)]
            τ_List[0] = [τ1]
            σ_fr[0] = DataImport.σ_fF / τ1
        else:
            τ_t2_temp = [BasicParams.SlipPlane(orList[N0 + item[1]], Δσ) for item in PnList[jj-1]]
            τ_t2 = [list(t) for t in zip(*τ_t2_temp)]

            σ_f0 = [σ_f_List[N0 + item[1]] for item in PnList[jj-1]]

            t_List = [ [] for _ in range(DataImport.Mg) ]
            τ_List = [ [] for _ in range(DataImport.Mg) ]
            σ_fr = [0] * DataImport.Mg

            t_List[0] = [TauT2_0 * np.outer(TauT2_1, TauT2_2) for TauT2_0, TauT2_1, TauT2_2 in zip(*τ_t2)] 
            τ_List[0] = list(τ_t2[0])

            σ_fr[0] = 1 / (RnAS[jj-1].dot([τ_t2[0][i] / σ_f0[i] for i in range(len(σ_f0))]))#问题
            τ1 = RnAS[jj-1].dot(τ_List[0])

        if σ_fr[0] < 1:
            cc = 1.05 * BasicParams.thickness
            Δδ = 0
            for m in range(len(rnList)):
                if cc < rnList[jj + m-1]:
                    Δδ = CTSD(aii, cc, m-1, σ_fr, τ1, rnList[jj-1:jj + m])
                    break

                elif jj + m +1  > len(Nd):
                    unstable = 1
                    goto=1
                    break

                N0 = sum(Nd[0 : jj+m])

                PnList_1_2 = np.array(PnList[jj + m-1])
                PnList1 = PnList_1_2[:, 0].tolist()
                PnList2 = PnList_1_2[:, 1].tolist()

                τ_t0_temp = [BasicParams.SlipPlane(orList[N0 + PnList2[i]], t_List[m][PnList1[i]]) for i in range(len(PnList1))]
                τ_t0 = [list(t) for t in zip(*τ_t0_temp)]
                τ0 = τ_t0[0]
                t0 = [tau_t0_0 * np.outer(tau_t0_1, tau_t0_2) for tau_t0_0, tau_t0_1, tau_t0_2 in zip(*τ_t0)] 

                def group_key(item):
                    return item[1]  
                
                grouped_data = groupby(PnList[jj + m - 1], key=group_key)
                SB = [list(group) for key, group in grouped_data]
                nSB = len(SB)
                t_gs = [ [] for _ in range(nSB) ]
                τ_gs = [ [] for _ in range(nSB) ]  

                for tt in range(nSB):
                    sb = SB[tt]
                    pos = [i for i, item in enumerate(PnList[jj + m - 1]) if item in sb]
                    rnas = sum(RnAS[jj + m - 1][pos[0]:pos[-1] + 1])

                    τ_gs[tt] = sum([τ0[i] * RnAS[jj + m-1][i] for i in pos]) / rnas

                    t_gs[tt] = sum([t0[i] * RnAS[jj + m-1][i] for i in pos]) / rnas

                σ_f0 = [σ_f_List[N0 + p[1]] for p in PnList[jj + m - 1]]
                σ_fr[m + 1] = 1 / (RnAS[jj + m - 1].dot([τ0[i] / σ_f0[i] for i in range(len(σ_f0))]))
                τ_List[m + 1] = τ_gs.copy()
                t_List[m + 1] = t_gs.copy()

                if σ_fr[m] < 1:
                    cc = 1.05 * BasicParams.thickness
                else:
                    cc = CCn(aii, m, σ_fr, rnList[jj - 1:jj + m])
        else:
            if i == 0:
                cc=100
                Δδ = 0
                goto=1
            else:
                cc = CC1(aii, σ_fr, rnList[0]) # calculate the length of the slip band in Stage I
                Δδ = 0
                for m in range(len(rnList)):
                    if cc < rnList[jj + m-1]:
                        Δδ = CTSD(aii, cc, m-1, σ_fr, τ1, rnList[jj-1:jj + m])
                        break

                    elif jj + m +1  > len(Nd):
                        unstable = 1
                        goto=1
                        break

                    N0 = sum(Nd[0 : jj+m])

                    PnList_1_2 = np.array(PnList[jj + m-1])
                    PnList1 = PnList_1_2[:, 0].tolist()
                    PnList2 = PnList_1_2[:, 1].tolist()

                    τ_t0_temp = [BasicParams.SlipPlane(orList[N0 + PnList2[i]], t_List[m][PnList1[i]]) for i in range(len(PnList1))]
                    τ_t0 = [list(t) for t in zip(*τ_t0_temp)]
                    τ0 = τ_t0[0]
                    t0 = [tau_t0_0 * np.outer(tau_t0_1, tau_t0_2) for tau_t0_0, tau_t0_1, tau_t0_2 in zip(*τ_t0)] 

                    def group_key(item):
                        return item[1]  
                    
                    grouped_data = groupby(PnList[jj + m - 1], key=group_key)
                    SB = [list(group) for key, group in grouped_data]
                    nSB = len(SB)
                    t_gs = [ [] for _ in range(nSB) ]
                    τ_gs = [ [] for _ in range(nSB) ]  

                    for tt in range(nSB):
                        sb = SB[tt]
                        pos = [i for i, item in enumerate(PnList[jj + m - 1]) if item in sb]
                        rnas = sum(RnAS[jj + m - 1][pos[0]:pos[-1] + 1])

                        τ_gs[tt] = sum([τ0[i] * RnAS[jj + m-1][i] for i in pos]) / rnas

                        t_gs[tt] = sum([t0[i] * RnAS[jj + m-1][i] for i in pos]) / rnas

                    σ_f0 = [σ_f_List[N0 + p[1]] for p in PnList[jj + m - 1]]
                    σ_fr[m + 1] = 1 / (RnAS[jj + m - 1].dot([τ0[i] / σ_f0[i] for i in range(len(σ_f0))]))
                    τ_List[m + 1] = τ_gs.copy()
                    t_List[m + 1] = t_gs.copy()

                    if σ_fr[m] < 1:
                        cc = 1.05 * BasicParams.thickness
                    else:
                        cc = CCn(aii, m, σ_fr, rnList[jj - 1:jj + m])
        # print(f"cc: {cc}")
        # print(f"Δδ: {Δδ}")
        # print(f"σ_fr: {σ_fr[:20]}")
        return cc, Δδ, unstable, goto


    def calc_σ(i, aii, rt1, y, z, ff00, σ_op_max):
        #----------------------------------------------sub_function_START------------------------------------------------
        def KWeight(at, ac):  # Define the function to calculate KWeight
            # Extract at, ac, and values from the list of lists
            if BasicParams.type in ["CS","CSS"]:
                if at <= 0.6:
                    interpolated_value = DataImport.WFrbf_low([[at, ac]])[0]
                elif at > 0.6:
                    interpolated_value = DataImport.WFrbf_high([[at, ac]])[0]
                return 1 / np.sqrt(BasicParams.thickness * at) * interpolated_value
            else:
            # Interpolate the value
                interpolated_value = DataImport.WFrbf([[at, ac]])[0]
                return 1 / np.sqrt(BasicParams.thickness * at) * interpolated_value
        

        def KWeight1(at, ac):  # Define the function to calculate KWeight
            # Extract at, ac, and values from the list of lists
            at_ac_values = [item[0] for item in DataImport.WFClist]
            values = [item[1] for item in DataImport.WFClist]

            # Flatten the at_ac_values and values lists
            at_values, ac_values = zip(*at_ac_values)
            at_values = np.array(at_values)
            ac_values = np.array(ac_values)
            values = np.array(values)
        # 
            mask_linear = at_values <= 0.1



            if np.any(mask_linear):
                interpolated_value = griddata(
                    np.column_stack((at_values[mask_linear], ac_values[mask_linear])),
                    values[mask_linear], (at, ac),method='linear'
                )
            return 1 / np.sqrt(BasicParams.thickness * at) * interpolated_value



        def KintegralI(aii, rt1, y, z): # Calculation of K-values via weight functions in the fisrt grain (Stage I)
            at_value = aii / BasicParams.thickness  
            ac_value = aii / rt1  
            a1, a2, a3, a4 = KWeight1(at_value, ac_value)

            WG1 = [[] for _ in range(6)]
            Results = np.empty(6)


            for i in range(6):
                WG1[i] =list(map(lambda x: (a1 + a2 * (1 / np.sqrt(1 - x)) + a3 * np.sqrt(1 - x) + a4 * (1 - x)) * 
                    (FieldValues.FieldValues_ACTIVE_Numpy((aii / 2) * (1 + x), y, z)[i]), DataImport.gp10) )
                
            for i in range(6):
                array_a = (aii / 2) * np.array(WG1[i])
                array_b = np.array(DataImport.gauss10)
                Results[i] = np.sum(array_a * array_b)

            return np.array(Results)


        def KintegralII(aii, asp, y, z): # Calculation of K-values via weight functions in the fisrt grain (Stage Ⅱ)            
            at_value = aii / BasicParams.thickness  

            a1, a2, a3, a4 = KWeight(at_value, asp)
            WG1 = [[] for _ in range(6)]
            Results = np.empty(6)


            for i in range(6):
                WG1[i] =list(map(lambda x: (a1 + a2 * (1 / np.sqrt(1 - x)) + a3 * np.sqrt(1 - x) + a4 * (1 - x)) *
                    (FieldValues.FieldValues_ACTIVE_Numpy((aii / 2) * (1 + x), y, z)[i]), DataImport.gp10) )

            for i in range(6):
                array_a = (aii / 2) * np.array(WG1[i])
                array_b = np.array(DataImport.gauss10)
                Results[i] = np.sum(array_a * array_b)

            return np.array(Results)



        def σ_remote_func(ai0,asp0): # Functions for converting to remote stresses
            if BasicParams.remote_type == 1:
                return DataImport.σ_CF(ai0 / BasicParams.thickness, asp0) * math.sqrt(ai0 / DataImport.QQ(asp0))
            else:
                return math.sqrt(math.pi * ai0)

        def Epsilon_Closure(σ_nom, σ_app, a, y, z, σ_op_max):
            def HysterisisA(y, z):
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = np.array(FieldValues.FieldValues_ACTIVE_Numpy(0, y, z))
                σ = np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                eigenvalues_sigma = np.linalg.eigvals(σ)
                return max(eigenvalues_sigma)

            def Hysterisis(σ_nom1, y, z):
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = np.array(FieldValues.FieldValuesAOpen(σ_nom1,0, y, z))
                σ = np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                eigenvalues_sigma = np.linalg.eigvals(σ)
                return max(eigenvalues_sigma)

            def HysterisisIto1(y, z):
                field_values = FieldValues.FieldValues_ACTIVE_Numpy(0, y, z)
                return max(field_values[3:6])
            
            def HysterisisIto2(σ_nom, y, z):
                field_values = FieldValues.FieldValuesAOpen(σ_nom,0, y, z)
                return max(field_values[3:6])            
            

            def OP(σ_max, R1): # Function returning the crack opening stress for any loading stress, full opening
                def OP00(σ_max, R1): 

                    αα = 3 # Plastic constraint factor 3, coefficient in plane strain state.
                    r_σ = σ_max / DataImport.σ_0 if σ_max <= DataImport.σ_0 else 1   
                    A0 = (0.825 - 0.34 * αα + 0.05 * αα**2) * (np.cos(np.pi / 2 * r_σ))**(1 / αα)
                    A1 = (0.415 - 0.071 * αα) * r_σ
                    A3 = 2 * A0 + A1 - 1
                    A2 = 1 - A0 - A1 - A3

                    if R1 >= 0:
                        σ_op = (A0 + A1 * R1 + A2 * R1**2 + A3 * R1**3) * σ_max
                    elif -1 <= R1 < 0:
                        σ_op = (A0 + A1 * R1) * σ_max
                    else:
                        σ_op = None

                    return σ_op
                
                if OP00(σ_max, R1) > σ_max * R1:  
                    return OP00(σ_max, R1)
                else:
                    return σ_max * R1
                 
            σ_max = σ_app/(1-BasicParams.r)
            σ_min = BasicParams.r * σ_max              #minium nominal stress
            σ_op = OP(σ_max, BasicParams.r)              #remote crack opening stress
            σ_op_max =σ_min + (σ_op - σ_min) * (1 - np.exp(-BasicParams.k_tr * a))
            if i<=3:
                return 1.0, σ_min, σ_min
            if σ_op_max > σ_min:
                ε_0 = Hysterisis(σ_nom,y, z)
                ε_1 = Hysterisis(( (σ_op_max - σ_min) /(σ_max - σ_min)*σ_nom), y, z)

                return (ε_0-ε_1)/ε_0, σ_op_max, σ_op_max
            else:
                return 1.0, σ_op_max, σ_op_max


        #----------------------------------------------sub_function_END------------------------------------------------
        if i > BasicParams.eval_num_stage1-1:
            if aii <= BasicParams.thickness * 0.95:
                asp = 1 / ff00(aii)
                K = KintegralII(aii, asp, y, z)
                σ_remote =  2*K / σ_remote_func(aii, asp)
                UU, σ_op, σ_op_max = Epsilon_Closure(BasicParams.σ_nom[load_quene], σ_remote[1], aii, y, z, σ_op_max)
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote
                Δσ = UU *np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                mark1111=1
            else:
                asp = 1 / ff00(0.95 * BasicParams.thickness)
                K = KintegralII(0.95 * BasicParams.thickness, asp, y, z)
                σ_remote =  2*K / σ_remote_func(aii, asp) 
                UU, σ_op, σ_op_max = Epsilon_Closure(BasicParams.σ_nom[load_quene], σ_remote[1],  0.95 * BasicParams.thickness, y, z, σ_op_max)
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote
                Δσ = UU * np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
#                print("crack>0.8*t:", "K=",K, "σ_remote =", σ_remote)
                mark1111=2
        else:

            if i > 0 :
                asp = aii / rt1
                K = KintegralI(aii, rt1, y, z)
                σ_remote =  2*K / σ_remote_func(aii, asp) 
                UU, σ_op, σ_op_max = Epsilon_Closure(BasicParams.σ_nom[load_quene], σ_remote[1], aii, y, z, σ_op_max)
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote
                Δσ = UU * np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                mark1111=3
            else :
                K=[0,0,0,0,0,0]
                asp = aii / rt1
                σ_remote =  2*np.array(FieldValues.FieldValues_ACTIVE_Numpy(0, y, z))
                UU, σ_op, σ_op_max = Epsilon_Closure(BasicParams.σ_nom[load_quene], σ_remote[1], aii, y, z, σ_op_max)
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote
                Δσ = UU * np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                mark1111=4


        if np.isnan(Δσ).any():
            print("σ_remote:",σ_remote)
            print("K:",K)
            print("rt1:",rt1)
            print("asp:",asp)
            print("aii:",aii)
            print("mark1111=",mark1111)  
        return asp, σ_remote[1], K[1], UU, Δσ , σ_op, σ_op_max     


    def get_EvalPoints(r1, rt1, rnList):

        if BasicParams. gb_effect== 0:
            ai = BasicParams.eval_points_full(r1)
            surfai = BasicParams.eval_points_full(r1)
            nai = len(ai)
        else:

            Ai1 = BasicParams.eval_points_stage1(r1)
            Ai1[-1] = r1 - DataImport.dave / 1000#

            rni0 = np.array([2, 3] + [1 + int(sum(np.heaviside(ai - rnList,0))) for ai in BasicParams.eval_points_stage2])-1
            rni1 = np.unique(rni0)
            #rni1 = np.arange(1, len(rnList) )
            rni = rni1[1:] if rni1[0] == 0 else rni1 

            Ai2 = np.array([[rnList[rn - 1] + DataImport.dave / 1000, 0.5 * (rnList[rn] + rnList[rn - 1]), rnList[rn] - DataImport.dave / 1000] for rn in rni]).flatten().tolist()
            ai = sorted(Ai1 + Ai2)
            nai = len(ai) 

            surfai = ai.copy() 
            surfai[:BasicParams.eval_num_stage1] = [2* rt1] * BasicParams.eval_num_stage1
        
        return ai, surfai, nai

    def evalCycle(i, cc, rnList, ai, nTemp, S_Δδ0, Scyc0Copy, dNdaListCopy, LifeMin):
        def NNf(x):
            if x == 0:
                return 10**18
            elif 0 < x < 1:
                return 1 + (-0.7 * math.log(x))**1.5
            else:
                return 1

        S_cyc0 = Scyc0Copy.copy()
        dNdaList = dNdaListCopy.copy()
        Label = 0

        # Stage I
        if i < BasicParams.eval_num_stage1:
            if i == 0:
                S_cyc0[i] = 0
                dNda = 0
            else:
                if S_Δδ0[i - 1] == 0 or S_Δδ0[i] == 0:
                    #print("**************Abort current calculation1************")
                    return S_cyc0, dNdaList

                dNda0 = 1 / (BasicParams.c_paris * S_Δδ0[i - 1]**BasicParams.n_paris)
                dNda1 = 1 / (BasicParams.c_paris * S_Δδ0[i]**BasicParams.n_paris)
                dNda = 0.5 * (dNda0 + dNda1)

                S_cyc0[i] = dNda * (ai[i] - ai[i - 1]) + S_cyc0[i - 1]
            NSuspend = NNf(S_cyc0[i] / nTemp)

            # To improve the calculation efficiency, the calculation is terminated when the number of cycles exceeds 10 times the minimum life
            if S_cyc0[i] > NSuspend * LifeMin(ai[i]):
                #print("**************Abort current calculation2************")
                Label = 0
                return S_cyc0, dNdaList , Label             

            if S_cyc0[i] > 10 * LifeMin(ai[i]):
                #print("**************Abort current calculation3************")
                Label = 0
                return S_cyc0, dNdaList, Label

            dNdaList[0] = [0, dNda]
            # print("dNdaList[0] : ",dNdaList[0])
        
        elif (i - BasicParams.eval_num_stage1 + 1) % 3 == 0:
            ai1, ai2 = ai[i - 1] - ai[i - 2], ai[i] - ai[i - 2]
            dNda0 = 1 / (BasicParams.c_paris * (S_Δδ0[i - 2]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris))
            dadN1 = 1 / (BasicParams.c_paris * (S_Δδ0[i - 1]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris)) - dNda0
            dadN2 = 1 / (BasicParams.c_paris * (S_Δδ0[i]**BasicParams.n_paris - BasicParams.Δδ_th**BasicParams.n_paris)) - dNda0

            if ai1 * ai2 * (ai1 - ai2) == 0:
                #print("**************Abort current calculation4************")
                return S_cyc0, dNdaList

            αα = (ai2 * dadN1 - ai1 * dadN2) / (ai1 * ai2 * (ai1 - ai2))
            ββ = (ai2**2 * dadN1 - ai1**2 * dadN2) / (ai1 * ai2 * (ai2 - ai1))
            γγ = dNda0

            dNda = (1 / ai2) * ((1/3) * αα * ai2**3 + (1/2) * ββ * ai2**2 + γγ * ai2)
            grNum = int((i -  BasicParams.eval_num_stage1 + 1) / 3)
            lineNum = int(sum(np.heaviside(ai[i] - rnListi, 0) for rnListi in rnList))
            dNdaList[grNum] = [lineNum, dNda]
            lineNumPrev, dNdaPrev = dNdaList[grNum - 1]
            rnPrev = rnList[lineNumPrev]

            S_cyc0[i - 2] = S_cyc0[i - 3] + dNdaPrev * (rnPrev - ai[i - 3]) + ((dNdaPrev + dNda) / 2) * (rnList[lineNum - 1] - rnPrev) + dNda * (ai[i - 2] - rnList[lineNum - 1])
            S_cyc0[i - 1] = S_cyc0[i - 2] + dNda * (ai[i - 1] - ai[i - 2])
            S_cyc0[i] = S_cyc0[i - 1] + dNda * (ai[i] - ai[i - 1])

            if min(S_cyc0[i - 2:i + 1]) > nTemp or min(S_cyc0[i - 2:i + 1]) < 0:
                Label = 1
                #print("**************Abort current calculation5************")
                return S_cyc0, dNdaList, Label

            for ii in range(i - 2, i + 1):
                if S_cyc0[ii] > nTemp:
                    Label = 1
                    #print("**************Abort current calculation6************")
                    return S_cyc0, dNdaList, Label
                NSuspend = NNf(S_cyc0[ii] / nTemp)

                if S_cyc0[ii] > NSuspend * LifeMin(ai[ii]):
                    Label = 1
                    #print("**************Abort current calculation7************")
                    return S_cyc0, dNdaList, Label

                if S_cyc0[ii] > 10 * LifeMin(ai[ii]):
                    Label = 1
                    #print("**************Abort current calculation8************")
                    return S_cyc0, dNdaList, Label

        return S_cyc0, dNdaList, Label


    dNdaList = [1 for _ in range(len(BasicParams.eval_points_stage2) + 10)]
    unstable = 0 #*unstable=1* if unstable destruction occurs
    skip = 0 #If the number of loadings is greater than the results for the other occurrence points skip = 1
    
    r1, rt1 = get_r1_rt1()
    ff00 = getCrackAspectFunc(r1, rt1, Anai)
    σ_f_List, rnList, Nd, PnList, RnAS, orList = generateGrains(r1, rt1, ff00)
    ai, surfai, nai = get_EvalPoints(r1, rt1, rnList)
    
    # Information initialization under breaking conditions
    S_Δδ0 = [0] * nai
    S_K0 = [0] * nai
    S_asp0 = [1] * nai
    S_RR0 = [BasicParams.r] * nai
    S_σ0 = [0] * nai
    S_c0 = [1.05 * BasicParams.thickness] * nai
    S_cyc0 = [10**18] * nai
    S_σ_op0 = [0] * nai
    σ_op_max = -10000
    # Calculate CTSD, slip zone length, and number of loading cycles for each crack length， then loop for life calculation.
    for i in range(nai):
        aii= ai[i]

        # ----------------------------------------------------------eval σ_remote,CTSD------------------------------------------------------------
        asp, σ_remote, K, UU, Δσ, σ_op, σ_op_max = calc_σ(i, aii, rt1, y, z, ff00, σ_op_max)

        cc, Δδ, unstable, goto = evalCTSD(aii, Δσ, σ_f_List, orList, rnList, Nd, PnList, RnAS)

        if goto == 1:
            break
    
        if unstable == 1:
            for j in range(i, nai):
                S_Δδ0[j] = float('inf')
                S_cyc0[j] = S_cyc0[j - 1] 
                S_c0[i] = float('inf') 
            break

        if BasicParams.eval_num_stage1-1< i and Δδ < BasicParams.Δδ_th:
            break

        # ----------------------------------------------------------evalCycle------------------------------------------------------------
        S_c0[i] = cc 
        S_Δδ0[i] = Δδ

        S_RR0[i] = (1 - BasicParams.r) * UU
        S_K0[i] = K

        S_σ0[i] = Δσ[1]
        S_σ_op0[i] = σ_op
        S_asp0[i] = 1 / asp if i > BasicParams.eval_num_stage1-1 else (0 if i == 0 else rt1 / aii)
        surfai[i] = 2 * S_asp0[i] * aii if i > BasicParams.eval_num_stage1-1 else 2 * rt1
        S_cyc0, dNdaList, Label = evalCycle(i, cc, rnList, ai, nTemp, S_Δδ0, S_cyc0, dNdaList, LifeMin)
        

        if Label == 1:
            # print("**************break-2************")
            break

        if cc >= BasicParams.thickness and (i - BasicParams.eval_num_stage1 + 1) % 3 == 0:
            cc0gr = int(sum(np.heaviside(BasicParams.thickness - S_c0i, 0) for S_c0i in S_c0)) -1
            Nf0 = S_cyc0[cc0gr]
            Nf2 = S_cyc0[cc0gr + 1]
            cc0 = S_c0[cc0gr]
            cc2 = S_c0[cc0gr + 1]
            Nf1 = S_cyc0[i] if cc == BasicParams.thickness else ((cc2 - BasicParams.thickness) * Nf0 + (BasicParams.thickness - cc0) * Nf2) / (cc2 - cc0)

            for iii in range(cc0gr + 1, nai):
                S_cyc0[iii] = Nf1
                S_Δδ0[iii] = S_Δδ0[cc0gr]
                S_c0[iii] = BasicParams.thickness
                S_K0[iii] = S_K0[cc0gr]
                S_asp0[iii] = S_asp0[cc0gr]
                S_RR0[iii] = S_RR0[cc0gr]
                S_σ0[iii] = S_σ0[cc0gr]
                
                S_σ_op0[iii] = S_σ_op0[cc0gr]


    return S_Δδ0, S_K0, S_asp0, S_RR0, S_σ0, S_c0, S_cyc0, ai, surfai, rnList, S_σ_op0
#Part5 End
#
#

    

# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------6. Life evaluation for single element------------------------------------------------------           
# -------------------------------------------------------------------------------------------------------------------------------------------    
def ElementLife(ia0, n, ia, σ, τ_max,FieldValues,S_cyc_result, S_ai_result):

# FatigueLife-------------------------------------------Initiate the frcture parameters--------------------------------------------------------
    S_yz = ["no data", "no data"] # location of crack initiation site
    S_ai = [0] * BasicParams.eval_num_total  # crack depth[mm]
    S_surf = [0] * BasicParams.eval_num_total # surface crack length [mm]
    S_Δδ = [0] * BasicParams.eval_num_total  # CTSD[mm]
    S_c = [0] * BasicParams.eval_num_total  # slip band length [mm]
    S_K = [0] * BasicParams.eval_num_total
    S_asp = [1] * BasicParams.eval_num_total
    S_RR = [BasicParams.r] * BasicParams.eval_num_total
    S_σ = [0] * BasicParams.eval_num_total
    S_cyc = [10**15] * BasicParams.eval_num_total  # number of loading cycles
    S_σ_op = [0] * BasicParams.eval_num_total
    rd = 0  # d/dmax
    RnList = None


#----------------------------------------------sub_function__START------------------------------------------------
    def makeFList(): # Definition of the function makeFList to generate the list of FirstGrain grain sizes, perlite short and long diameters
        Aej = BasicParams.active_element_area # area of area element Ae
        d = 0 # flag for loop termination
        fww = 25
        n_max= round(fww*DataImport.ngAe)
        FList0 = [[] for _ in range(n_max)]  # List of FirstGrain grain sizes, perlite short and long diameters

        i = 0
        while d != 1:
            PorF = random.random()
            if PorF > DataImport.PRateN:
                i += 1
                fd = DataImport.FirstGrainCDF(random.random()) 

                if Aej < (np.pi * fd**2) / 4:
                    fd = np.sqrt((4 * Aej) / np.pi)
                    d = 1
                FList0.append(fd)
                Aej -= (np.pi * fd**2) / 4
            else:
                r = random.random()
                pw = DataImport.SecondGrainCDF(r)
                pl = DataImport.FirstGrainCDF(r)

                if Aej < np.pi / 4 * pw * pl:
                    pw = (4 * Aej) / (np.pi * pl)
                    d = 1

                Aej -= np.pi / 4 * pw * pl

        FList = [sublist for sublist in FList0 if sublist]
        sorted_FList = sorted(FList, reverse=True)
        return sorted_FList


    #generateFoList
    def generateFoList():
        FoList = [BasicParams.makeEulerAngles() for _ in range(FNum)]#
        return FoList
    
    #Anai method for crack shape calculation
    def prepareAnai(y, z):
        def prepare_sigma_grad(y, z):# calculate the gradient of the stress field on surface
            xx = np.arange(0, BasicParams.thickness * 0.9, 0.01)
            σ_grad0 = [(x/BasicParams.thickness, 2 * FieldValues.FieldValues_ACTIVE_Numpy(x, y, z)[1]) for x in xx]#normalize
            coeff = np.polyfit([item[0] for item in σ_grad0], [item[1] for item in σ_grad0], 19)
            return coeff

        coeffs= prepare_sigma_grad(y, z)
        sigma_grad = np.poly1d(coeffs)

        def f1_derivative(xx):
            derivative_coefficients = np.polyder(sigma_grad).coeffs
            first_derivative = np.poly1d(derivative_coefficients)
            return first_derivative(xx)

        def Anai0(xx):
            return -0.06 + 0.47 * 0.5 * (f1_derivative(0.5*xx)*0.5 + f1_derivative(xx)) / sigma_grad((0))

        def Anai(xx):
            xx_vals = (xx * (1 + DataImport.gp10)) / 2
            BB = Anai0(xx_vals)
            return 1 + (xx / 2) * np.dot(BB, DataImport.gauss10)  
        return Anai



#----------------------------------------------sub_function__END------------------------------------------------

    y = BasicParams.AeYZ[ia][0]
    z = BasicParams.AeYZ[ia][1]

    FList = makeFList()
    FNum = len(FList)  
    fNum = np.sum(np.heaviside(np.array(FList) - BasicParams.grain_size_lim * DataImport.dmax, 0))

    FoList = generateFoList()
    Anai = prepareAnai(y, z)
    if τ_max > DataImport.σ_fF and fNum > 0:
#Life evaluation loop for single element----------------------------------------------Area Element---------------------------------------------------------------
        for Grain_i in range(int(fNum)):
            τ1, θn1, θs1 = BasicParams.SlipPlane(FoList[0], σ)
            if τ1 > DataImport.σ_fF:
                S_Δδ0, S_K0, S_asp0, S_RR0, S_σ0, S_c0, S_cyc0, ai, surfai, rnList0, S_σ_op0 = CrackLifeCalc(FList[Grain_i], FoList[Grain_i], y, z, Anai,FieldValues,S_cyc_result, S_ai_result)

                if S_cyc0[-1] < S_cyc_result[-1]:

                    S_yz = [y + BasicParams.active_element_size[0] * (random.random() - 0.5), z + BasicParams.active_element_size[1] * (random.random() - 0.5)]

                    S_ai = ai  
                    S_surf = surfai
                    S_Δδ = S_Δδ0 
                    S_K = S_K0 
                    S_asp = S_asp0 
                    S_RR = S_RR0
                    S_σ = S_σ0
                    S_c = S_c0 
                    S_cyc = S_cyc0
                    S_cyc_result = S_cyc0
                    S_σ_op = S_σ_op0
                    rd = FList[Grain_i]/ DataImport.dmax
                    RnList = rnList0
                    print(f"  Ae No. {(ia0) * BasicParams.n_symm + n+1}  Grain: {Grain_i + 1}/{FNum} ({fNum})  d/dmax={rd}  " + f"  N={S_cyc[-1]}" + f" yz={S_yz}"  )

    else:
        print(f"  Ae No. {(ia0) * BasicParams.n_symm + n+1}  fatigue crack will not initiate in this area element")        
    return S_yz, S_cyc, S_ai, S_surf, S_c, S_Δδ, S_K, S_σ, S_RR, S_asp, S_σ_op, RnList
#Part6 End
#




# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------7. Execution of the model---------------------------------------------------------------            
# -------------------------------------------------------------------------------------------------------------------------------------------            
#1.import basic parameters
BasicParams = BasicParameters() # Basic parameters

#2.Determining if an abaqus database has been created and import the data
pkl_file_name =BasicParams.steel + "_"+ BasicParams.type + "_FieldValues.pkl"
if os.path.exists(pkl_file_name):
    DataImport = MaterialDataImporter(BasicParams)
else:
    AbaqusDatabaseCreator(BasicParams) # Create an abaqus database
    DataImport = MaterialDataImporter(BasicParams) # Import abaqus and material database

#3. Create a dictionary to store the FieldValues
FieldValuesDict = {}
for load_quene in range(len(BasicParams.σ_nom)):
    FieldValuesDict[load_quene] = FieldValuesFunction(BasicParams, DataImport, load_quene)

#4. Run the main function
def main(FieldValues, load_quene):

    #----------------------------------------------sub_function__START------------------------------------------------
    def stressAE( i,FieldValues):#Stress tensor of AREA ELEMENT at a specific stress
        y = BasicParams.AeYZ[i, 0]#element y coordinate
        z = BasicParams.AeYZ[i, 1]#element z coordinate
        # Surface stress tensor definition
        σ = np.array(FieldValues.FieldValues_ACTIVE_Numpy(0, y, z))
        if np.isnan(σ[0]):
            return []
        else:
            σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = (2*σ).tolist() 
            Δσ = np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])

            eigenvalues = np.linalg.eigvals(Δσ)
            τ_max = (max(eigenvalues) - min(eigenvalues)) / 2.0
            return list([i, Δσ, τ_max])
    #----------------------------------------------sub_function__END------------------------------------------------
        

    stressAE_list0 = [stressAE(i, FieldValues) for i in range(BasicParams.AeNum)]
    stressAE_list1 = [sub_list for sub_list in stressAE_list0 if sub_list]
    iστ = sorted(stressAE_list1, key=lambda x: x[2], reverse=True)

    iAE = [item[0] for item in iστ]
    σ_AE = [item[1] for item in iστ]
    τ_max_AE = np.array([item[2] for item in iστ])

    imax = round(BasicParams.AeNum*BasicParams.stress_lim)
    print("imax :",imax)


    total_results = []  # Number of calculations
    total_results1 = []

    for iteration in range(len(BasicParams.iteration_num)):
        min_S_cyc = 10**15
        S_ai_result = None
        weak_S_yz= None
        min_element_index = None
        excel_data = []
        eval_point_num = 0
        S_yz_sub = ["no data", "no data"] 
        S_ai_sub = [0] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2)
        S_cyc_sub = [10**15] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2) 
        S_cyc_result = [10**15] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2)  
        S_ai_result = [0] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2) 
        RnList_result = [0]
        RnList_sub = None

        nocrack = 0
        for i in range(imax):#
            print("Element", i + 1, "Total:", imax)  # 
            for n in range(BasicParams.n_symm):#
                print("  Symmetry", n + 1, "Total:", BasicParams.n_symm)  #
                weak_S_yz_i, S_cyc_result_i, S_ai_result_i, S_surf_i, S_c_i, S_Δδ_i, S_K_i, S_σ_i, S_RR_i, S_asp_i,S_σ_op_i, RnList_i = ElementLife(n, i, iAE[i], σ_AE[i], τ_max_AE[i],FieldValues,S_cyc_sub, S_ai_sub)

                if S_cyc_result_i[-1] < S_cyc_sub[-1]:
                    S_cyc_sub = S_cyc_result_i
                    S_yz_sub = weak_S_yz_i
                    S_ai_sub = S_ai_result_i
                    S_surf_sub = S_surf_i
                    S_c_sub = S_c_i
                    S_Δδ_sub = S_Δδ_i
                    S_K_sub = S_K_i
                    S_σ_sub = S_σ_i
                    S_RR_sub = S_RR_i
                    S_asp_sub = S_asp_i
                    S_σ_op_sub = S_σ_op_i
                    RnList_sub = RnList_i



            if S_cyc_sub[-1] < min_S_cyc: #mark the minimum life
                min_S_cyc = S_cyc_sub[-1]
                weak_S_yz = S_yz_sub
                S_cyc_result = S_cyc_sub
                S_ai_result = S_ai_sub
                S_surf_result = S_surf_sub
                S_c_result = S_c_sub
                S_Δδ_result = S_Δδ_sub
                S_K_result = S_K_sub
                S_σ_result = S_σ_sub
                S_RR_result = S_RR_sub
                S_asp_result = S_asp_sub
                S_σ_op_result = S_σ_op_sub
                RnList_result = RnList_sub
                nocrack = 1


                row_data = [i + 1] + list(S_yz_sub) + list(S_cyc_sub)
                excel_data = [[BasicParams.iteration_num[iteration] + 1]+row_data]

        total_results.extend(excel_data)
        if weak_S_yz is not None:
            print("  Nf=", S_cyc_result[-1], "   (y,z)=(", weak_S_yz[0], ",", weak_S_yz[1], ")  ", datetime.datetime.now())
        eval_point_num0=len(S_cyc_result) 
        if eval_point_num0 > eval_point_num:
            eval_point_num = eval_point_num0
        
        total_results1.append([BasicParams.σ_nom[load_quene], BasicParams.iteration_num[iteration] + 1,  min_S_cyc, weak_S_yz])
        max_len = max(
            len(S_ai_result) ,len(RnList_result)
        )
        def pad_list(lst, target_len):
            if isinstance(lst, np.ndarray):
                lst = lst.tolist()
            return lst + [np.nan] * (target_len - len(lst))
        if nocrack == 1:
            data = {
                "Crack depth": pad_list(S_ai_result, max_len),
                "Surface crack length": pad_list(S_surf_result, max_len),
                "Slip band": pad_list(S_c_result, max_len),
                "CTSD": pad_list(S_Δδ_result, max_len),
                "Cycles": pad_list(S_cyc_result, max_len),
                "K Value": pad_list(S_K_result, max_len),
                "Effective Stress": pad_list(S_σ_result, max_len),
                "Stress Ratio": pad_list(S_RR_result, max_len),
                "Aspect Ratio": pad_list(S_asp_result, max_len),
                "Opening Stress": pad_list(S_σ_op_result, max_len),
                "Location of GB": pad_list(RnList_result, max_len)
            }

            df = pd.DataFrame(data)

            # Constructing filename based on your template
            filename = f"{BasicParams.name}_{BasicParams.n_paris}_{BasicParams.Δδ_th}_{BasicParams.σ_nom[load_quene]}_{DataImport.σ_fF}_{BasicParams.iteration_num[iteration]}.{BasicParams.r}.csv"

            # Exporting to CSV
            df.to_csv(filename, index=False)

        
    # save results to csv file
    # column_names = ['Iteration'] + ['Element', 'Y', 'Z'] + [f'S_cyc_{i}' for i in range(1, eval_point_num + 1)] + ['minimum life']
    # data_with_columns = [column_names] + total_results
    # df = pd.DataFrame(data_with_columns)
    # csv_filename = f'{BasicParams.name}_{BasicParams.n_paris}_{BasicParams.Δδ_th}_{BasicParams.σ_nom[load_quene]}MPa.csv' 
    # df.to_csv(csv_filename, index=False) 

    column_names1 = ['σ_nom', 'Iteration', 'minimum life', 'yz']
    df1 = pd.DataFrame(total_results1, columns=column_names1)
    csv_filename1 = f'{BasicParams.name}_{BasicParams.n_paris}_{BasicParams.Δδ_th}_{BasicParams.σ_nom[load_quene]}_{DataImport.σ_fF}.{BasicParams.r}.csv'
    df1.to_csv(csv_filename1, index=False)





if __name__ == "__main__":
    for load_quene in range(len(BasicParams.σ_nom)):
        print("load_quene : ",BasicParams.σ_nom[load_quene], datetime.datetime.now())
        FieldValues=FieldValuesDict[load_quene]
        main(FieldValues,load_quene)
