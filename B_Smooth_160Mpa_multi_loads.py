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


###############################################STRUCTURE OF THE MODEL###############################################
# 1. Basic information import (class: BasicParameters)
# 2. ABAQUS data import (class: AbaqusDatabaseCreator)
# 3. Material data import (class: MaterialDataImporter)
# 4. Creat Field value function (class: FieldValuesFunction)
# 5. Life evaluation for single crack (fuction: CrackLifeCalc)
# 6. Life evaluation for sigle area element (fuction: ElementLife)
# 7. Global functions
# 8. Execution of the model
###############################################STRUCTURE OF THE MODEL###############################################





# -------------------------------------------------------------------------------------------------------------------------------------------  
# ----------------------------------------------------------1. Basic information import--------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------  

class BasicParameters:
    def __init__(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Initialize all parameter
        self.name = os.path.basename(os.getcwd()) # The name of the specimen
        self.steel = self.name.split("_")[:2][0] # The steel type of the specimen
        self.type = self.name.split("_")[:2][1] # The type of the test
        self.σ_nom = np.arange(135,176,5)#Applied stress
        self.n_symm = 4 #Number of model calculations (typically 4 for 1/4 model)

        # Parameters related to fissure aperture ratio
        self.closure_type = 2 # The closure type of the specimen
        self.r = -1 if self.type == "Smooth" else (0.09 if self.type in ["R008", "R100"] else (-1 if type == "CS" else None)) # SressRatio
        self.k_tr = 2 * 14 #Material constant for opening stress calculation
        self.gb_effect = 1 #Grain boundary effect type
        self.remote_type = 1 #Remote stress conversion type

        #life calculation
        self.c_paris = 11.8 #Paris law coefficient
        self.n_paris = 2.0 #Paris law exponent
        self.Δδ_th = 0.000145 #threshold for crack-tip sliding displacement range

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
        self.width = 5 if self.type in ["Smooth", "R008", "R100"] else 1.5
        self.thickness = 1.5 if (self.type == "Smooth" and (self.steel == "Bainite" or self.steel == "Martensite")) else \
            (1.35 if self.type == "Smooth" else (1.9 if self.type in ["R008", "R100"] else 4.5))

    #Active zone defination
    def active_zone(self):
        if self.type == "Smooth":
            self.y_lim = 1.0
            self.z_lim = 2.5
            self.grain_size_lim = 0.2
            self.active_element_size = [0.125, 0.125]
            self.stress_lim = 0.5

        elif self.type == "R100":
            self.y_lim = 0.5
            self.z_lim = 2.5
            if self.steel == "B":
                self.grain_size_lim = 0.2
                self.active_element_size = [0.125, 0.125]
                self.stress_lim = 0.85
            elif self.steel == "N50R":
                self.grain_size_lim = 0.4
                self.active_element_size = [0.05, 0.125]
                self.stress_lim = 0.97
            elif self.steel == "Bainite":
                self.grain_size_lim = 0.6
                self.active_element_size = [0.05, 0.125]
                self.stress_lim = 0.95

        elif self.type == "R008":
            self.y_lim = 0.04
            self.z_lim = 2.5
            self.grain_size_lim = 0.2
            self.active_element_size = [0.02, 0.05]
            self.stress_lim = 0.85

        elif self.type == "CS":
            self.y_lim = 0.025
            self.z_lim = 0.75
            self.grain_size_lim = 0.2
            self.active_element_size = [0.025, 0.15]
            if self.steel == "E":
                self.stress_lim = 0.85
            elif self.steel == "N50R":
                self.stress_lim = 0.95
        
        self.active_element_area = self.active_element_size[0] * self.active_element_size[1]
        
        # Creation of area element coordinates
        self.AeNumY = int(self.y_lim / self.active_element_size[0])
        self.AeNumZ = int(self.z_lim / self.active_element_size[1])
        self.AeNum = self.AeNumY * self.AeNumZ  # Number of area elements in the target area
        self.AeYZ = np.array([[self.active_element_size[0] * (i - 0.5), self.active_element_size[1] * (j - 0.5)]
                              for i in range(1, self.AeNumY + 1)for j in range(1, self.AeNumZ + 1)])


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

            if self.steel == "Bainite":
                self.eval_points_stage2 = {
                    "Smooth": [0.2, 0.6, 1.1],
                    "R100": [0.035, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.4, 0.7, 1.2],
                    "R008": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 1.5]
                }.get(self.type)

            elif self.steel == "Martensite":
                self.eval_points_stage2 = {
                    "Smooth": [0.2, 0.6, 1.1],
                    "R100": [0.035, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.4, 0.7, 1.2],
                    "R008": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 1.5]
                }.get(self.type)

            elif self.steel in ["B", "E"]:
                self.eval_points_stage2 = {
                    "Smooth": [0.2, 0.6, 1.1],
                    "R100": [0.2, 0.4, 0.7, 1.2],
                    "R008": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 1.5]
                }.get(self.type)

            elif self.steel == "CS":
                self.eval_points_stage2 = {
                    "Smooth": [0.2, 0.6, 1.1],
                    "R100": [0.2, 0.4, 0.7, 1.2],
                    "R008": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 1.5],
                    "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                           1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                           3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4]
                }.get(self.type)

            elif self.steel == "N50R":
                self.eval_points_stage2 = {
                    "Smooth": [0.08, 0.2, 0.6, 1.1],
                    "R100": [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.4, 0.7, 1],
                    "R008": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 1.5],
                    "CS": [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, \
                           1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, \
                           3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4]      
                }.get(self.type)
        self.eval_num_stage2 = len(self.eval_points_stage2) + 2 #改
        self.eval_num_total = self.eval_num_stage1 + self.eval_num_stage2
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
        self.AA = self.E / (4 * π * (1 - self.ν ** 2))

    #Model information
    def read_inp(self):
        #----------------------------------------------sub_function_START------------------------------------------------
        def move_instance(inst, xyz0): # Definition of the function for transforming the coordinates of nodes.
            tr = np.array(inst[0][:3]) 
            xyz1 =xyz0+tr
            a0 = np.array(inst[1][:3])   
            a1 = np.array(inst[1][3:6])  
            axis = (a1 - a0) / np.linalg.norm(a1 - a0) 
            phi = π * inst[1][-1] / 180
            def rformula(xyz):
                r = xyz - a0
                return r * np.cos(phi) + axis * (np.dot(axis, r)) * (1 - np.cos(phi)) + np.cross(axis, r) * np.sin(phi)
            xyz2 = [list(map(rformula,  xyz1))]  
            return xyz2+a0
        
        def x0(y, z): 
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
        instance_data_lines = [line.strip() for line in self.inp_data[instance_start_index:end_instance_index]]
        instance = [list(map(float, line.split(','))) for line in instance_data_lines]  
        if instance[0] == ["*End Instance"]:
            instance = [[0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]
        if instance[1] == ["*End Instance"]:
            instance = [instance[0], [1, 0, 0, 0, 0, 0, 0]]     
        self.node = np.round(move_instance(instance, self.node0)[0], 4)

        # Definition of the nodal set Triming in the Active Zone
        eTrimStart = [i for i, item in enumerate(self.inp_data) if "elset=Trim" in item][0] + 1
        if self.inp_data[eTrimStart - 1].split(',')[-1]==" generate\n": 

            eTrim0 = np.array([list(range(int(self.inp_data[eTrimStart].split(',')[0]), int(self.inp_data[eTrimStart].split(',')[1]) + 1, int(self.inp_data[eTrimStart].split(',')[2])))]) 
        else:
            eTrimStart_index = l_symbol.index(eTrimStart-1) + 1        
            eTrimEnd = l_symbol[eTrimStart_index]-1
            eTrim_lines = self.inp_data[eTrimStart:eTrimEnd + 1]
            eTrim0 = np.array([list(map(int, line.split(",")[0:16])) for line in eTrim_lines]) 

        eTrim = (eTrim0.flatten()-1).tolist() 
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
        
        elif self.BasicParams.type == "CS":
            Load_Lines0 = [idx+2 for idx, line in enumerate(self.inp_data) if "*Dsload" in line]
            Load_lines = [sublist for index, sublist in enumerate(self.inp_data) if index + 1 in Load_Lines0]
            sigma_app_edge = [-1 * float(line.split(",")[2]) for line in Load_lines]
            sigma_nom_list0 = [(1.5 * 5) / (self.BasicParams.width * self.BasicParams.thickness) * σ for σ in sigma_app_edge]

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
            'DataEES': self.DataEES,
            'DataPES': self.DataPES,
            'DataAES': self.DataAES,
            'FDataE': self.FDataE,
            'FDataP': self.FDataP,
            'FDataA': self.FDataA, 
            'EES': self.EES,
            'PES': self.PES,
            'AES': self.AES    
            }

        # Specify the file name for saving 
        pkl_file_name =self.BasicParams.steel + "_"+ self.BasicParams.type + "_FieldValues.pkl"

        # Save the data to a file
        with open(pkl_file_name, 'wb') as file:
            pickle.dump(data_to_save, file)

        return print("Complete Definition of FieldValues Function :", datetime.now())
#Part2 End
#
#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------3. Material data import --------------------------------------------------------------            
# -------------------------------------------------------------------------------------------------------------------------------------------            

class MaterialDataImporter:
    def __init__(self, BasicParams):
        self.BasicParams = BasicParams


        # Import perlite grain size from csv file
        self.FerriteDF = pd.read_csv(self.BasicParams.steel + "_Ferrite grain size.csv", header=None).to_numpy()
        self.FerriteDFID = np.unique(self.FerriteDF[:, 0])
        self.ferriteMax = self.FerriteDF[-1]
        self.FerriteAspectDF = pd.read_csv(self.BasicParams.steel + "_Ferrite grain aspect ratio.csv", header=None).to_numpy()

        # Import perlite grain size from csv file
        self.pearlite_thickness_files = [file for file in os.listdir() if file.startswith(self.BasicParams.steel + "_Pearlite thickness") and file.endswith(".csv")]
        self.PearliteDF = pd.read_csv(self.pearlite_thickness_files[0], header=None).to_numpy()

        # distribution fraction
        self.PearliteCDF = makeCDF(self.PearliteDF)
        self.FerriteCDF = makeCDF(self.FerriteDF)
        self.FerriteCDFr = makeCDFr(self.FerriteDF)
        self.PearliteCDFr = makeCDFr(self.PearliteDF)
        self.FerriteAspectCDFr = makeCDFrA(self.FerriteAspectDF)

        # Ferrite average and maximum grain size
        self.dave = sum(item[0] ** 3 * item[1] for item in self.FerriteDF) / sum(item[0] ** 2 * item[1] for item in self.FerriteDF)
        self.dmax = self.FerriteCDF(1)
        self.ngAe = (4 * self.BasicParams.active_element_area) / (π * self.dave**2)

        # Import Ferrite grain aspect ratio distribution from csv file
        self.FerriteAspectDF = pd.read_csv(self.BasicParams.steel + "_Ferrite grain aspect ratio.csv", header=None).to_numpy()

        #size of the model
        self.Ng = None
        self.Mg = None

        # Pearlite fraction
        self.PRate = None

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
        self.values = None
        self.gauss10 = np.array([0.06667134430869041, 0.14945134915057556, 0.21908636251598135, 0.2692667193099942, 0.2955242247147529, 0.2955242247147529, 0.26926671930999607, 0.21908636251598135, 0.1494513491505782, 0.06667134430869041])
        self.gp10 = np.array([-0.9739065285171692, -0.8650633666889868, -0.679409568299024, -0.4333953941292474, -0.14887433898163122, 0.14887433898163122, 0.43339539412924727, 0.679409568299024, 0.8650633666889848, 0.9739065285171692])
        #Remote stress
        self.CFlist = None

        #
        self.gData = None
        self.PRateN = None

        #call fuctions
        self.pearlite_fraction()
        self.monotonic_tensile_properties()
        self.friction_strength()
        self.abaqus_data_read()
        self.weight_function_constants()
        self.remote_stress_constants()
        self.CreateGrainData()
        self.ModelSize()


    #Volume fraction of pearlite
    def pearlite_fraction(self):
        pattern = re.compile(r"(\d+\.\d+)") 
        match = re.search(pattern, str(self.pearlite_thickness_files))
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
        self.σ_fF, self.σ_fP = 2*friction_strength_data.iloc[0, [0, 1]]        

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
        self.DataEES = data['DataEES']
        self.DataPES = data['DataPES']
        self.DataAES = data['DataAES']
        self.FDataE = data['FDataE']
        self.FDataP = data['FDataP']
        self.FDataA = data['FDataA']
        self.EES = data['EES']
        self.PES = data['PES']
        self.AES = data['AES']
        print("Complete Reading ABAQUS FieldValues Function from pkl File :", datetime.now())
    

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
        self.sec_ell = interpolate.interp1d(x_values, y_values, kind='linear')

        # Calculation of K-values via weight functions
        if self.BasicParams.type == "Smooth":
            self.WFClist = WFCsmooth
        elif self.BasicParams.type == "R008" or BasicParams.type == "R100":
            self.WFClist = WFC3PBT
        elif self.BasicParams.type == "CS":
            self.WFClist = WFCCS
    
    
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

        if self.BasicParams.type == "Smooth":
            self.CFlist = CFsmooth
        else:
            self.CFlist = CF3PBT    

        # Calculate CFlist2
        CFlist2 = [((row[0], row[1]), (row[2] * np.sqrt(self.QQ(row[1]))) / np.sqrt(row[0] * self.BasicParams.thickness)) for row in self.CFlist]
        x_values1, y_values = zip(*CFlist2) 
        x_values = np.array(x_values1)
        self.σ_CF = LinearNDInterpolator(x_values, y_values) 



        




    def CreateGrainData(self):
        def VolumeFraction():#
            if self.BasicParams.steel == "Bainite" or self.BasicParams.steel == "Martensite":
                self.PRateN = 0
            else:
                # Generating Random Numbers
                VP = np.mean([self.PearliteCDF(r) * self.FerriteCDF(r) for r in [random.random() for _ in range(100000)]])
                VF = np.mean([self.FerriteCDF(random.random()) ** 2 for _ in range(100000)])
                self.PRateN = (self.PRate / VP) / (self.PRate / VP + (1 - self.PRate) / VF)   
        VolumeFraction()

        def makegList(r, d, t, ra):
            if r > self.PRateN:
                return np.array([self.σ_fF, ra * d, d / ra]).tolist()
            else:
                return  np.array([self.σ_fP, ra * d, t]).tolist()    

        if os.path.exists(self.BasicParams.steel + "_"+ self.BasicParams.type + "_gData.pkl"):
            with open(self.BasicParams.steel + "_"+ self.BasicParams.type + "_gData.pkl", "rb") as file:
                self.gData = pickle.load(file)
                print("Finish loading gData.pkl", datetime.now())
        else:
            print("Start generating gData.pkl", datetime.now())
            nData = 1000000
            R = np.random.rand(3, nData) 
            dtrList = [[row[0], self.FerriteCDFr(row[2]), self.PearliteCDFr(row[2]), self.FerriteAspectCDFr(row[1])] for row in R.T]

            self.gData = list(map(lambda x: makegList(*x), dtrList))#Generate a list of nData grains (containing ferite and pearlite)
            print("Finish generating gData.pkl", datetime.now())

            with open(self.BasicParams.steel + "_"+ self.BasicParams.type + "_gData.pkl", "wb") as file:
                pickle.dump((self.gData), file)


    def ModelSize(self):#   Model size
        # Generate the first part of A0
        part1 = [(self.FerriteCDF(random.random()))**2 for i in range(int(50000 * (1 - self.PRateN)))]
        # Generate the second part of A0
        part2 = [self.FerriteCDF(random.random()) * self.PearliteCDF(random.random()) for i in range(int(50000 * self.PRateN))]
        # Combine the two parts to create A0
        A0 = part1 + part2

        fw = 4 if self.BasicParams.steel == "Smooth" else 6
        self.Ng = round(fw * ((2 * self.BasicParams.thickness**2) /np.mean(A0))) # Ng is the sufficient large number of grains in the area of fracture surface

        d0 = np.mean([self.FerriteCDF(random.random()) * self.FerriteAspectCDFr(random.random()) for i in range(50000)])
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
# -------------------------------------------------------------------------------------------------------------------------------------------
    def FieldValuesA(self,x, y, z):
        s = self.DataImport.f_sigma(self.BasicParams.σ_nom[self.load_queue])
        s1 = int(np.floor(s))
        s2 = int(np.ceil(s))

        if s - s1 <= 0.1:
            result = [self.DataImport.AES[k][s1](x, y, z) for k in range(6)]
        elif s2 - s <= 0.1:
            result = [self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
        else:
            result = [(s2 - s) * self.DataImport.AES[k][s1](x, y, z) + (s - s1) * self.DataImport.AES[k][s2](x, y, z) for k in range(6)]

        return result
    
    def FieldValuesAOpen(self,σ, x, y, z):
        s = self.DataImport.f_sigma(σ)
        s1 = int(np.floor(s))
        s2 = int(np.ceil(s))

        if s - s1 <= 0.1:
            result = [self.DataImport.AES[k][s1](x, y, z) for k in range(6)]
        elif s2 - s <= 0.1:
            result = [self.DataImport.AES[k][s2](x, y, z) for k in range(6)]
        else:
            result = [(s2 - s) * self.DataImport.AES[k][s1](x, y, z) + (s - s1) * self.DataImport.AES[k][s2](x, y, z) for k in range(6)]

        return result


    #generate x_Present for 1D interpolation
    def x_Present_creator(self):
        start = 0
        end = 1.9

        # Define the initial step size and growth factor
        initial_step = 0.01
        step_growth_factor = 1

        # Initialize an empty list to store data
        self.x_Present = []

        # Gradually increase the step size and generate data
        current_step = initial_step
        current_value = start

        while current_value < end:
            self.x_Present.append(current_value)
            current_value += current_step
            current_step *= step_growth_factor

        self.x_Present.append(end)
        self.x_Present = np.array(self.x_Present)
        self.x_Present_number=len(self.x_Present)


    # Definition of the function FDataPAE to calculate the stress and strain of each area element
    def FData_AE(self):

        AeNumY = int(self.BasicParams.y_lim / self.BasicParams.active_element_size[0])
        AeNumZ = int(self.BasicParams.z_lim / self.BasicParams.active_element_size[1])

        AeNum = AeNumY * AeNumZ  # Number of area elements in the target area

        # Calculate the centre coordinates of each area element
        self.AeYZ = np.array([[self.BasicParams.active_element_size[0] * (i - 0.5), self.BasicParams.active_element_size[1] * (j - 0.5)]
                    for i in range(1, AeNumY + 1)
                    for j in range(1, AeNumZ + 1)])

        y_Present = [[self.AeYZ[i, 0]] * self.x_Present_number for i in range(len(self.AeYZ))]
        z_Present = [[self.AeYZ[i, 1]] * self.x_Present_number for i in range(len(self.AeYZ))]
        stress_active0 = [self.FieldValuesA(self.x_Present, y_Present[i], z_Present[i])for i in range(len(self.AeYZ))]
        self.stress_active_ele=np.array(stress_active0)


    def FieldValues_ACTIVE_Numpy(self,x, y, z):
        for i in range(len(self.AeYZ)):
            if self.AeYZ[i, 0] == y and self.AeYZ[i, 1] == z:
                location = i
                break

        return [np.interp(x, self.x_Present,self.stress_active_ele[location][k])for k in range(6)]
#Part4 End
#
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
        aspg = DataImport.FerriteAspectCDFr(random.random())
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
            if a0 == 0:
                return F2(a)**(-1/3)
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
                    return b * asp1 * DataImport.sec_ell(asp1)
                else:
                    return b * DataImport.sec_ell(1/asp1)
            else:
                return BasicParams.width  
    
#----------------------------------------------sub_function__END-----------------------------------------------
        g_List = np.array([[DataImport.σ_fF, r1, rt1]] + [random.choice(DataImport.gData) for _ in range(DataImport.Ng)])
        σ_f_List, dnList, tnList = g_List[:, 0], g_List[:, 1], g_List[:, 2]

        orList = [or1] + [makeEulerAngles() for _ in range(DataImport.Ng)]# 改

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

            if rr < BasicParams.thickness * 0.9:
            
                rr0 = rr + np.dot(dnList[m0 : m02+1], tnList[m0 : m02+1]) / (2 * np.sum(tnList[m0 : m02+1]))

                if L2(ff00(rr0), rr0) < np.sum(tnList[m0:m02+1]):
                    nnn = 1

                    while (m02 - 3 * nnn) >= m0:
                        rr00 = rr + np.dot(dnList[m0 : m02 - 3 * nnn + 1], tnList[m0 : m02 - 3 * nnn + 1]) / (2 * np.sum(tnList[m0 : m02 - 3 * nnn + 1]))

                        if L2(ff00(rr00), rr00) > np.sum(tnList[m0 : m02 - 3 * nnn+1]):
                            break
                        nnn += 1
                    gr = 1
                
                else:
                    nnn = 1
                    rr00 = rr + np.dot(dnList[m0 : m02 + 3 * nnn + 1], tnList[m0 : m02 + 3 * nnn + 1]) / (2 * np.sum(tnList[m0 : m02 + 3 * nnn + 1]))
                    while (m02 + 3 * nnn) <= DataImport.Ng and L2(ff00(rr00), rr00) > np.sum(tnList[m0:m02 + 3 * nnn+1]):
                        nnn += 1  
                        rr00 = rr + np.dot(dnList[m0 : m02 + 3 * nnn + 1], tnList[m0 : m02 + 3 * nnn + 1]) / (2 * np.sum(tnList[m0 : m02 + 3 * nnn + 1]))
                    gr = 2
                
            else:
                rr0 = rr + np.dot(dnList[m0:m02+1], tnList[m0:m02+1]) / (2 * np.sum(tnList[m0:m02+1]))

                if L2(ff00(0.9*BasicParams.thickness), rr0) < np.sum(tnList[m0 : m02+1]):
                    nnn = 1
                    while (m02 - 3 * nnn+1) >= m0:
                        if L2(ff00(0.9*BasicParams.thickness), rr + np.dot(dnList[m0:m02 - 3 * nnn+1], tnList[m0:m02 - 3 * nnn+1]) / (2 * np.sum(tnList[m0:m02 - 3 * nnn+1]))) > np.sum(tnList[m0:m02 - 3 * nnn+1]):
                            break
                        nnn += 1
                    gr = 1

                else:
                    nnn = 1
                    while (m02 + 3 * nnn+1) <= DataImport.Ng and L2(ff00(0.9*BasicParams.thickness), rr + np.dot(dnList[m0:m02 + 3 * nnn+1], tnList[m0:m02 + 3 * nnn+1]) / (2 * np.sum(tnList[m0:m02 + 3 * nnn+1]))) > np.sum(tnList[m0:m02 + 3 * nnn+1]):
                        nnn += 1
                    gr = 2

            n0 = n1 + 1
            n1 = m02 - 3 * nnn + 3 if gr == 1 else m02 + 3 * nnn - 3
            if n1 > DataImport.Ng:
                break

            Nd[n + 1] = n1 - n0 + 1

            RnA[n + 1] = tnList[n0:n1+1] / sum(tnList[n0:n1+1])
            RnAA[n + 1] = np.cumsum(RnA[n + 1])
            rnList[n + 1] = rnList[n] + np.dot(RnA[n + 1], dnList[n0:n1+1])
            k0, k1 = 1, 1
            Pn = [0] * (Nd[n] + Nd[n + 1] - 1)
            Pn[0] = [0, 0]
            Pn[Nd[n] + Nd[n + 1] - 2] = [Nd[n], Nd[n + 1]]

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
            c_root = brentq(Eq_1, a + 1e-10, rnlist+50000, args=(a, σ_fr))
            return c_root


        def CCn(a, n, σ_fr, rnList):
            def Eq_n(c, a, n, σ_fr, rnList):
                epsilon = 1e-10 # Avoid dividing by zero
                return np.pi * 0.5 - σ_fr[0] * safe_arccos(a / (c + epsilon)) - sum((σ_fr[i+1] - σ_fr[i]) * safe_arccos(rnList[i] / (c + epsilon)) for i in range(n+1))
            c_root = brentq(Eq_n, a + 1e-10, rnList[n]+50000, args=(a, n, σ_fr, rnList))
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
            result = Δτ_j / (π ** 2 * DataImport.AA) * (term1 + term2)
            return result
        #----------------------------------------------sub_function__END------------------------------------------------

        unstable = 0
        goto=0

        rnList_temp = rnList.tolist()
        jj = rnList_temp.index(next(filter(lambda x: x > aii, rnList_temp)))
        N0 = sum(Nd[0 : jj + 1]) - 1 

        E_σ =  np.linalg.eigvals(Δσ)
        τ_max=(max(E_σ)-min(E_σ))/2
        if jj == 0:
            τ1, θn1, θs1 = SlipPlane(orList[0], Δσ)

            t_List = [ [] for _ in range(DataImport.Mg) ]
            τ_List = [ [] for _ in range(DataImport.Mg) ]
            σ_fr = [0] * DataImport.Mg
            t_List[0] = [τ1 * np.outer(θn1, θs1)]
            τ_List[0] = [τ1]
            σ_fr[0] = DataImport.σ_fF / τ1
        else:
            τ_t2_temp = [SlipPlane(orList[N0 +1+ item[1]], Δσ) for item in PnList[jj]]
            τ_t2 = [list(t) for t in zip(*τ_t2_temp)]

            σ_f0 = [σ_f_List[N0 +1+ item[1]] for item in PnList[jj]]

            t_List = [ [] for _ in range(DataImport.Mg) ]
            τ_List = [ [] for _ in range(DataImport.Mg) ]
            σ_fr = [0] * DataImport.Mg

            t_List[0] = [TauT2_0 * np.outer(TauT2_1, TauT2_2) for TauT2_0, TauT2_1, TauT2_2 in zip(*τ_t2)] 
            τ_List[0] = list(τ_t2[0])

            σ_fr[0] = 1 / (RnAS[jj].dot([τ_t2[0][i] / σ_f0[i] for i in range(len(σ_f0))]))
            τ1 = RnAS[jj].dot(τ_List[0])
       
        if σ_fr[0] < 1:
            cc = 1.05 * BasicParams.thickness
            Δδ = 0
            for m in range(len(rnList)):
                if cc < rnList[jj + m]:
                    Δδ = CTSD(aii, cc, m-1, σ_fr, τ_max, rnList[jj:jj + m ])
                    break

                elif (jj + 1) + (m + 1) > len(Nd):
                    unstable = 1
                    goto=1
                    break

                N0 += Nd[jj + m + 1]

                PnList_1_2 = np.array(PnList[jj + m])
                PnList1 = PnList_1_2[:, 0].tolist()
                PnList2 = PnList_1_2[:, 1].tolist()

                τ_t0_temp = [SlipPlane(orList[N0+1 + PnList2[i]], t_List[m][PnList1[i]]) for i in range(len(PnList1))]
                τ_t0 = [list(t) for t in zip(*τ_t0_temp)]
                τ0 = τ_t0[0]
                t0 = [tau_t0_0 * np.outer(tau_t0_1, tau_t0_2) for tau_t0_0, tau_t0_1, tau_t0_2 in zip(*τ_t0)] 

                def group_key(item):
                    return item[1]  
                
                grouped_data = groupby(PnList[jj + m], key=group_key)
                SB = [list(group) for key, group in grouped_data] 
                nSB = len(SB)
                t_gs = [ [] for _ in range(nSB) ]
                τ_gs = [ [] for _ in range(nSB) ]  

                for tt in range(nSB):
                    sb = SB[tt]
                    pos = [i for i, item in enumerate(PnList[jj + m]) if item in sb]
                    rnas = sum(RnAS[jj + m][pos[0]:pos[-1] + 1])

                    τ_gs[tt] = sum([τ0[i] * RnAS[jj + m][i] for i in pos]) / rnas

                    t_gs[tt] = sum([t0[i] * RnAS[jj + m][i] for i in pos]) / rnas

                σ_f0 = [σ_f_List[N0+1 + p[1]] for p in PnList[jj + m]]
                σ_fr[m + 1] = 1 / (RnAS[jj + m].dot([τ0[i] / σ_f0[i] for i in range(len(σ_f0))]))
                τ_List[m + 1] = τ_gs.copy()
                t_List[m + 1] = t_gs.copy()

                if σ_fr[m+1] < 1:
                    cc = 1.05 * BasicParams.thickness
                else:
                    cc = CCn(aii, m, σ_fr, rnList[jj:jj + m + 2])
        else:
            if i == 0:
                cc=100
                Δδ = 0
                goto=1
            else:
                cc = CC1(aii, σ_fr, rnList[jj]) # calculate the length of the slip band in Stage I
                Δδ = 0
                for m in range(len(rnList)):
                    if cc < rnList[jj + m]:
                        Δδ = CTSD(aii, cc, m-1, σ_fr, τ_max, rnList[jj:jj + m ])
                        break

                    elif (jj + 1) + (m + 1) > len(Nd):
                        unstable = 1
                        break

                    N0 += Nd[jj + m + 1]

                    PnList_1_2 = np.array(PnList[jj + m])
                    PnList1 = PnList_1_2[:, 0].tolist()
                    PnList2 = PnList_1_2[:, 1].tolist()

                    τ_t0_temp = [SlipPlane(orList[N0+1 + PnList2[i]], t_List[m][PnList1[i]]) for i in range(len(PnList1))]


                    τ_t0 = [list(t) for t in zip(*τ_t0_temp)]
                    τ0 = τ_t0[0]
                    t0 = [tau_t0_0 * np.outer(tau_t0_1, tau_t0_2) for tau_t0_0, tau_t0_1, tau_t0_2 in zip(*τ_t0)] 


                    def group_key(item):
                        return item[1]  

                    grouped_data = groupby(PnList[jj + m], key=group_key)
                    SB = [list(group) for key, group in grouped_data] 
                    nSB = len(SB)
                    t_gs = [ [] for _ in range(nSB) ]
                    τ_gs = [ [] for _ in range(nSB) ]  

                    for tt in range(nSB):
                        sb = SB[tt]
                        pos = [i for i, item in enumerate(PnList[jj + m]) if item in sb]
                        rnas = sum(RnAS[jj + m][pos[0]:pos[-1] + 1])

                        τ_gs[tt] = sum([τ0[i] * RnAS[jj + m][i] for i in pos]) / rnas
                        t_gs[tt] = sum([t0[i] * RnAS[jj + m][i] for i in pos]) / rnas


                    σ_f0 = [σ_f_List[N0+1 + p[1]] for p in PnList[jj + m]]
                    σ_fr[m + 1] = 1 / (RnAS[jj + m].dot([τ0[i] / σ_f0[i] for i in range(len(σ_f0))]))
                    τ_List[m + 1] = τ_gs.copy()
                    t_List[m + 1] = t_gs.copy()

                    if σ_fr[m] < 1:
                        cc = 1.05 * BasicParams.thickness
                    else:
                        cc = CCn(aii, m, σ_fr, rnList[jj:jj + m + 2])

        if Δδ < BasicParams.Δδ_th:
            goto=1
        return cc, Δδ, unstable, goto


    def calc_σ(i, aii, rt1, y, z, ff00):
        #----------------------------------------------sub_function_START------------------------------------------------
        def KWeight(at, ac): # Define the function to calculate KWeight

            # Extract at, ac, and values from the list of lists
            at_ac_values = [item[0] for item in DataImport.WFClist]
            values = [item[1] for item in DataImport.WFClist]

            # Flatten the at_ac_values and values lists
            at_values, ac_values = zip(*at_ac_values)
            at_values = np.array(at_values)
            ac_values = np.array(ac_values)
            values = np.array(values)

            # Define the points for interpolation
            points = np.column_stack((at_values, ac_values))

            interpolated_value = griddata(points, values, (at, ac), method='linear')
            return 1 / np.sqrt(BasicParams.thickness * at) * interpolated_value        


        def KintegralI(aii, rt1, y, z): # Calculation of K-values via weight functions in the fisrt grain (Stage I)
            at_value = aii / BasicParams.thickness  
            ac_value = aii / rt1  
            a1, a2, a3, a4 = KWeight(at_value, ac_value)

            WG1 = [[] for _ in range(6)]
            Results = np.empty(6)

           
            for i in range(6):
                WG1[i] =list(map(lambda x: (a1 + a2 * (1 / np.sqrt(1 - x)) + a3 * np.sqrt(1 - x) + a4 * (1 - x)) * (
                    FieldValues.FieldValues_ACTIVE_Numpy((aii / 2) * (1 + x), y, z)[i]), DataImport.gp10) )
                
            for i in range(6):
                array_a = (aii / 2) * np.array(WG1[i])
                array_b = np.array(DataImport.gauss10)
                Results[i] = np.sum(array_a * array_b)

            return np.array(Results[1])
        
               
        def KintegralII(aii, asp, y, z): # Calculation of K-values via weight functions in the fisrt grain (Stage Ⅱ)            
            at_value = aii / BasicParams.thickness  

            a1, a2, a3, a4 = KWeight(at_value, asp)
            WG1 = [[] for _ in range(6)]
            Results = np.empty(6)

            for i in range(6):
                WG1[i] =list(map(lambda x: (a1 + a2 * (1 / np.sqrt(1 - x)) + a3 * np.sqrt(1 - x) + a4 * (1 - x)) *
                 (FieldValues.FieldValues_ACTIVE_Numpy((aii / 2) * (1 + x), y, z)[i]),DataImport.gp10) )

            for i in range(6):
                array_a = (aii / 2) * np.array(WG1[i])
                array_b = np.array(DataImport.gauss10)
                Results[i] = np.sum(array_a * array_b)

            return np.array(Results[1])
        


        def σ_remote_func(ai0,asp0): # Functions for converting to remote stresses
            if BasicParams.remote_type == 1:
                return DataImport.σ_CF(ai0 / BasicParams.thickness, asp0) * math.sqrt(ai0 / DataImport.QQ(asp0))
            else:
                return math.sqrt(math.pi * ai0)


        def Epsilon_Closure(σ_nom, σ_app, R1, a, y, z):
            def HysterisisA(y, z):
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = np.array(FieldValues.FieldValues_ACTIVE_Numpy(0, y, z))
                σ = np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                eigenvalues_sigma = np.linalg.eigvals(σ)
                return max(eigenvalues_sigma)

            def Hysterisis(σ_nom, y, z):
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = np.array(FieldValues.FieldValuesAOpen(σ_nom,0, y, z))
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
                    A0 = (0.825 - 0.34 * αα + 0.05 * αα**2) * (np.cos(π / 2 * r_σ))**(1 / αα)
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
                
                if σ_max < DataImport.σ_0:
                    if OP00(σ_max, R1) > σ_max * R1:  
                        return OP00(σ_max, R1)
                    else:
                        return σ_max * R1
                else:
                    return σ_max * R1            

            σ_op = OP(σ_app, BasicParams.r)              #crack opening stress
            σ_min = BasicParams.r * σ_app                #minium nominal stress
            σ_optr =σ_min + (σ_op - σ_min) * (1 - np.exp(-BasicParams.k_tr * a))


            if BasicParams.closure_type == 1:
                return (σ_app - σ_optr) / (σ_app - σ_min)
            else:

                ε_0 = HysterisisIto1(y, z)
                ε_1 = HysterisisIto2(( σ_nom[load_quene]*(σ_optr - σ_min) / σ_app/(1-BasicParams.r)), y, z)

                if σ_optr<=σ_min:
                    return 1.0  
                else:
                    return (ε_0-ε_1)/ε_0

            
        #----------------------------------------------sub_function_END------------------------------------------------
        σ_00= FieldValues.FieldValues_ACTIVE_Numpy(0, y, z)
        σ_surf=σ_00/np.abs(σ_00[1])

        if i > BasicParams.eval_num_stage1-1:
            if aii <= BasicParams.thickness * 0.8:
                asp = 1 / ff00(aii)

                K = KintegralII(aii, asp, y, z)
                σ_remote =  2*K / σ_remote_func(aii, asp) /(1-BasicParams.r)
                UU = Epsilon_Closure(BasicParams.σ_nom, σ_remote, BasicParams.r, aii, y, z)
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote*σ_surf
                Δσ = UU * (1-BasicParams.r)*np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])

            else:
                asp = 1 / ff00(0.8 * BasicParams.thickness)
                K = KintegralII(0.8 * BasicParams.thickness, asp, y, z)
                σ_remote =  2*K / σ_remote_func(0.8 * BasicParams.thickness, asp) /(1-BasicParams.r)
                UU = Epsilon_Closure(BasicParams.σ_nom, σ_remote, BasicParams.r, 0.8 * BasicParams.thickness, y, z)
                
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote*σ_surf
                Δσ = UU *  (1-BasicParams.r)*np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
        else:
            asp = aii / rt1
            UU = 1
            if i > 0 and aii >= BasicParams.thickness * 0.001:
                K = KintegralI(aii, rt1, y, z)
                σ_remote =  2*K / σ_remote_func(aii, asp)  /(1-BasicParams.r)

                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_remote*σ_surf
                Δσ = (1-BasicParams.r)*np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
            
            elif i > 0 and aii < BasicParams.thickness * 0.001:
                K=0
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_00
                σ_remote =  2*σ_00[1]  /(1-BasicParams.r)
                Δσ = (1-BasicParams.r)*np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])

            elif i == 0:
                σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ_00
                σ_remote =  2*σ_00[1]  /(1-BasicParams.r)
                Δσ = (1-BasicParams.r)*np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])
                K=0

        return asp, σ_remote, K, UU, Δσ        


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
                return 10**15
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

            if min(S_cyc0[i - 2:i + 1]) > nTemp:
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
    S_cyc0 = [10**15] * nai

    # Calculate CTSD, slip zone length, and number of loading cycles for each crack length， then loop for life calculation.
    for i in range(nai):
        aii= ai[i]

        # ----------------------------------------------------------eval σ_remote,CTSD------------------------------------------------------------
        asp, σ_remote, K, UU, Δσ = calc_σ(i, aii, rt1, y, z, ff00)
        cc, Δδ, unstable,goto = evalCTSD(aii, Δσ, σ_f_List, orList, rnList, Nd, PnList, RnAS)

        if goto == 1:
            break
    
        if unstable == 1:
            for j in range(i, nai):
                S_Δδ0[j] = float('inf')
                S_cyc0[j] = S_cyc0[j - 1] 
                S_c0[i] = float('inf') 
            break

        if BasicParams.eval_num_stage1 < i and Δδ < BasicParams.Δδ_th:
            break

        # ----------------------------------------------------------evalCycle------------------------------------------------------------
        S_c0[i] = cc 
        S_Δδ0[i] = Δδ

        S_RR0[i] = (1 - BasicParams.r) * UU
        S_K0[i] = K

        S_σ0[i] = σ_remote

        S_asp0[i] = 1 / asp if i > BasicParams.eval_num_stage1 else (0 if i == 0 else rt1 / aii)
        surfai[i] = 2 * S_asp0[i] * aii if i > BasicParams.eval_num_stage1 else 2 * rt1
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


    return S_Δδ0, S_K0, S_asp0, S_RR0, S_σ0, S_c0, S_cyc0, ai, surfai
#Part5 End
#
#

    

# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------6. Life evaluation for single element------------------------------------------------------           
# -------------------------------------------------------------------------------------------------------------------------------------------    
def ElementLife(n, ia0, ia, σ, τ_max,FieldValues,S_cyc_result, S_ai_result):

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
    rd = 0  # d/dmax


#----------------------------------------------sub_function__START------------------------------------------------
    def makeFList(): # Definition of the function makeFList to generate the list of ferrite grain sizes, perlite short and long diameters
        Aej = BasicParams.active_element_area # area of area element Ae
        d = 0 # flag for loop termination
        fww = 20  # weigth
        n_max= round(fww*DataImport.ngAe)
        FList0 = [[] for _ in range(n_max)]  # List of ferrite grain sizes, perlite short and long diameters

        i = 0
        while d != 1:
            PorF = random.random()
            if PorF > DataImport.PRateN:
                i += 1
                fd = DataImport.FerriteCDF(random.random()) 

                if Aej < (np.pi * fd**2) / 4:
                    fd = np.sqrt((4 * Aej) / np.pi)
                    d = 1
                FList0.append(fd)
                Aej -= (np.pi * fd**2) / 4
            else:
                r = random.random()
                pw = DataImport.PearliteCDF(r)
                pl = DataImport.FerriteCDF(r)

                if Aej < np.pi / 4 * pw * pl:
                    pw = (4 * Aej) / (np.pi * pl)
                    d = 1

                Aej -= np.pi / 4 * pw * pl

        FList = [sublist for sublist in FList0 if sublist]
        sorted_FList = sorted(FList, reverse=True)
        return sorted_FList


    #generateFoList
    def generateFoList():
        FoList = [makeEulerAngles() for _ in range(FNum)]#
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
            τ1, θn1, θs1 = SlipPlane(FoList[0], σ)
            if τ1 > DataImport.σ_fF:
                S_Δδ0, S_K0, S_asp0, S_RR0, S_σ0, S_c0, S_cyc0, ai, surfai = CrackLifeCalc(FList[Grain_i], FoList[Grain_i], y, z, Anai,FieldValues,S_cyc_result, S_ai_result)

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
                    rd = FList[Grain_i]/ DataImport.dmax
                    print(f"  Ae No. {(ia0) * BasicParams.n_symm + n+1}  Grain: {Grain_i + 1}/{FNum} ({fNum})  d/dmax={rd}  " + f"  N={S_cyc[-1]}")

    else:
        print(f"  Ae No. {(ia0) * BasicParams.n_symm + n+1}  fatigue crack will not initiate in this area element")        
    return S_yz, S_cyc, S_ai
#Part6 End
#
#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------7. global functions---------------------------------------------------------------            
# -------------------------------------------------------------------------------------------------------------------------------------------            
global SlipPlane, π, makeCDF, makeCDFr, makeCDFrA, makeEulerAngles

π = np.pi
def SlipPlane(ang, DeltaSigma):
    # ang: Euler angles of the grain
    # DeltaSigma: stress tensor range

    nv = np.array([[[1., 1., 0.], [1., -1., 1.], [-1., 1., 1.]],
                [[1., -1., 0.], [1., 1., 1.], [-1., -1., 1.]],
                [[1., 0., 1.], [1., 1., -1.], [-1., 1., 1.]],
                [[-1., 0., 1.], [-1., 1., -1.], [1., 1., 1.]],
                [[0., 1., 1.], [1., 1., -1.], [1., -1., 1.]],
                [[0., 1., -1.], [1., 1., 1.], [1., -1., -1.]]])
    C_Psi, C_Theta, C_Phi = np.cos(ang)
    S_Psi, S_Theta, S_Phi = np.sin(ang)
        
    g_Psi_Theta_Phi = np.array([                            # Rotation matrix
        [C_Psi*C_Theta*C_Phi - S_Psi*S_Phi, S_Psi*C_Theta*C_Phi + C_Psi*S_Phi, -S_Theta*C_Phi],
        [-C_Psi*C_Theta*S_Phi - S_Psi*C_Phi, -S_Psi*C_Theta*S_Phi + C_Psi*C_Phi, S_Theta*S_Phi],
        [C_Psi*S_Theta, S_Psi*S_Theta, C_Theta]
        ])

    nv2 = np.array([list(map(lambda x: g_Psi_Theta_Phi.dot(x), sublist)) for sublist in nv])# 

    tau_list = []
    for i in range(6):
        for j in range(2):
            tau = abs(np.dot(np.dot(nv2[i, 0], DeltaSigma), nv2[i, j + 1]))
            tau_list.append((tau, i, j))
    
    tau_list.sort(key=lambda x: x[0], reverse=True)
    i0, j0 = tau_list[0][1], tau_list[0][2]
    
    max_shear_stress = 1.0 / np.sqrt(6) * tau_list[0][0] 
    max_principal_stress_dir = 1.0 / np.sqrt(2) * np.sign(nv2[i0, 0, 0]) * nv2[i0, 0]
    max_theta_n = 1.0 / np.sqrt(3) * np.sign(nv2[i0, 0, 0]) * nv2[i0, j0 + 1]

    return max_shear_stress, np.array(max_principal_stress_dir), np.array(max_theta_n)


def makeCDF(DF): #  Definition of the function makeCDF to calculate the cumulative distribution function of particle size
    CDF0 = np.cumsum(DF[:, 1])
    CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
    return interp1d(CDF, DF[:, 0], kind='linear', fill_value='extrapolate')

def makeCDFr(DF): #  Definition of the function makeCDF to calculate the cumulative distribution function of particle size
    CDF0 = np.cumsum(DF[:, 1])
    CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
    return interp1d(CDF, np.sqrt(π) / 2. * DF[:, 0], kind='linear', fill_value='extrapolate')

def makeCDFrA(DF):
    CDF0 = np.cumsum(DF[:, 1])
    CDF = CDF0 * 0.99999 + np.arange(len(DF)) / len(DF) * 0.00001
    RA = np.sqrt(DF[:, 0])
    return interp1d(CDF, RA, kind='linear', fill_value='extrapolate')

def makeEulerAngles():#
    return ([random.uniform(0, 2 * π), math.acos(random.uniform(-1, 1)), random.uniform(0, 2 * π)])

def stressAE(σ_nom, i,FieldValues):#Stress tensor of AREA ELEMENT at a specific stress
    y = BasicParams.AeYZ[i, 0]#element y coordinate
    z = BasicParams.AeYZ[i, 1]#element z coordinate
    # Surface stress tensor definition
    σ = np.array(FieldValues.FieldValues_ACTIVE_Numpy(0, y, z))
    if np.isnan(σ[0]):
        return []
    else:
        σ_11, σ_22, σ_33, τ_12, τ_13, τ_23 = σ.tolist()    
        Δσ = np.array([[σ_11, τ_12, τ_13], [τ_12, σ_22, τ_23], [τ_13, τ_23, σ_33]])

        eigenvalues = np.linalg.eigvals(Δσ)
        τ_max = (max(eigenvalues) - min(eigenvalues)) / 2.0
        return list([i, Δσ, τ_max])
#Part7 End
#
#



# -------------------------------------------------------------------------------------------------------------------------------------------            
# ----------------------------------------------------------8. Execution of the model---------------------------------------------------------------            
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

    # FatigueLife--------------------------------------------calculate stress for each AE--------------------------------------------------------
    stressAE_list0 = [stressAE(BasicParams.σ_nom[load_quene], i, FieldValues) for i in range(BasicParams.AeNum)]
    stressAE_list1 = [sub_list for sub_list in stressAE_list0 if sub_list]
    iστ = sorted(stressAE_list1, key=lambda x: x[2], reverse=True)

    iAE = [item[0] for item in iστ]
    σ_AE = [item[1] for item in iστ]
    τ_max_AE = np.array([item[2] for item in iστ])

    τmax_AEi = max(τ_max_AE) * BasicParams.stress_lim
    print("τmax_AEi :",τmax_AEi)

    imax = round(BasicParams.AeNum*BasicParams.stress_lim)
    print("imax :",imax)


    total_results = []  # Number of calculations
    for iteration in range(4):
        min_S_cyc = float('inf')
        S_ai_result = None
        weak_S_yz= None
        min_element_index = None
        excel_data = []
        S_yz_sub = ["no data", "no data"] 
        S_ai_sub = [0] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2)
        S_cyc_sub = [10**15] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2) 
        S_cyc_result = [10**15] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2)  
        S_ai_result = [0] * (BasicParams.eval_num_stage1+3*BasicParams.eval_num_stage2) 
        for i in range(imax):#
            print("Element", i + 1, "Total:", imax)  # 
            for n in range(BasicParams.n_symm):#
                weak_S_yz_i, S_cyc_result_i, S_ai_result_i = ElementLife(n, i, iAE[i], σ_AE[i], τ_max_AE[i],FieldValues,S_cyc_result, S_ai_result)
                if S_cyc_result_i[-1] < S_cyc_sub[-1]:
                    S_cyc_sub = S_cyc_result_i
                    S_yz_sub = weak_S_yz_i
                    S_ai_sub = S_ai_result_i
            if S_cyc_sub[-1] < min_S_cyc: #mark the minimum life
                min_S_cyc = S_cyc_sub[-1]
                weak_S_yz = S_yz_sub
                S_cyc_result = S_cyc_sub
                S_ai_result = S_ai_sub
                min_element_index = i
                row_data = [i + 1] + list(S_yz_sub) + list(S_cyc_sub)
                excel_data.append([iteration + 1]+row_data)
        for row in excel_data:
            row.append("YES" if row[1] == min_element_index + 1 else "  ")
        total_results.extend(excel_data)
        print("  Nf=", S_cyc_result[-1], "   (y,z)=(", weak_S_yz[0], ",", weak_S_yz[1], ")  ", datetime.now())
    # save results to csv file
    eval_point_num=len(S_cyc_result)
    column_names =['Iteration'] + ['Element', 'Y', 'Z'] + [f'S_cyc_{i}' for i in range(1, eval_point_num + 1)] + ['minimum life']
    df = pd.DataFrame(total_results, columns=column_names)
    excel_filename = f'results_{BasicParams.σ_nom[load_quene]}MPa.xlsx'
    df.to_excel(excel_filename, index=False)

if __name__ == "__main__":
    for load_quene in range(len(BasicParams.σ_nom)):
        print("load_quene : ",BasicParams.σ_nom[load_quene], datetime.now())
        FieldValues=FieldValuesDict[load_quene]
        main(FieldValues,load_quene)