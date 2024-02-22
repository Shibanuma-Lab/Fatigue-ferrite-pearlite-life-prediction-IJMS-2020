# Overview
This code predicts the fatigue life and limit of steels using microstructure information, tensile properties, and loading conditions.

# Requirements
* **Abaqus**: Required for generating .dat files.
  * You can either use the generated .dat file directly or import the .inp file into Abaqus to generate a .dat file.
 
* **Python 3.10**: Necessary for running the simulation. Download Python 3.10 from [here](https://www.python.org/downloads/).
  * The following libraries are also required: `os`, `re`, `math`, `random`, `pickle`, `numpy`, `pandas`, `datetime`, and `scipy`. Install these libraries using the command:
    `pip install os re math random pickle numpy pandas datetime scipy`

# Input Files Description
* **B_Ferrite grain aspect ratio.csv**: Ferrite grain aspect ratio distribution for steel B.
* **B_Ferrite grain size.csv**: Ferrite grain size distribution for steel B.
* **B_Friction strength.csv**: Friction strength required to move dislocations for steel B.
* **B_Monotonic tensile test.csv**: Static tensile properties for steel B.
* **B_Pearlite thickness, C=0.18.csv**: Pearlite grain size distribution and carbon mass fraction for steel B.
* **B_Smooth.inp**: Abaqus model information for specimen B_Smooth.
* **B_Smooth.dat**: Abaqus calculation results for specimen B_Smooth.
* **CombinedData_Ellipse_WF_CF.json**: Ellipse shape function.

# Structure of the Code
1. **Basic Information Import** (class: `BasicParameters`)
   * Defines model parameters including applied loads, Paris' law constants, model size, evaluation points, etc.
2. **ABAQUS Data Import** (class: `AbaqusDatabaseCreator`)
   * Imports Abaqus data, including .inp and .dat files.
3. **Material Data Import** (class: `MaterialDataImporter`)
   * Imports material parameters, such as microstructural information and tensile properties.
4. **Create Field Value Function** (class: `FieldValuesFunction`)
   * Defines a FieldValues function to obtain model strain data.
5. **Life Evaluation for Single Crack Initiation Site** (function: `CrackLifeCalc`)
6. **Life Evaluation for Single Area Element** (function: `ElementLife`)
7. **Execution of the Model**

# How to Execute
1. Place all files in a directory named according to the steel type and test type, e.g., `SteelType_TestType`.
   * For example, if the Steel Type is B and the Test Type is Smooth, the directory name should be `B_Smooth`.
2. Modify the values in the **class `BasicParameters`**.

3. Execute `MultiScale.py`. The fatigue life and crack location data will be exported to an output file named **Results_xxMPa.xlsx**. For extracting additional data evaluated during the simulation, slight modifications to `B_Smooth.py` may be necessary.
