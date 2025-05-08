# Overview
This code is designed to predict fatigue life and limit of steels based on microstructure information, tensile properties, and loading conditions.

# Requirements
* [**Abaqus**]: This is required to generate .dat file.
  * You can use the generated .dat file or import the .inp file into Abaqus to generate .dat file.
 
* [**Python 3.10**](https://www.python.org/downloads/): This is necessary for running the simulation.
  * Additionally, the following libraries are required: os, re, math, Random, pickle, numpy, pandas, datetime and scipy. You can install these libraries using the following commands:
    `pip install os, re, math, Random, pickle, numpy, pandas, datetime and scipy`

# Structure of the code
1. Basic information import (class: BasicParameters)
  * Define the model parameters including applied loads, Paris'law constants, model size,        evaluation points...
2. ABAQUS data import (class: AbaqusDatabaseCreator)
  * Import the abaqus data, including .inp data and .dat data.
3. Material data import (class: MaterialDataImporter)
  * Import material parameters, including microstructural information and tensile properties.
4. Creat Field value function (class: FieldValuesFunction)
  * Defining FieldValues function to obtain model strain data.
5. Life evaluation for single crack (fuction: CrackLifeCalc)
6. Life evaluation for sigle area element (fuction: ElementLife)
7. Execution of the model

# How to Execute
1. Put all the files in the same directory called Steel type_Test type.
   * In the sample file given, the Steel Type is B and the Test type is Smooth, so the directory name is B_Smooth.
2. Modify the values in the **class BasicParameters**.

3. Execute **B_Smooth.py**. The fatigue life and crack location data will be exported in the output file named **Resulys_xxMPa.xlsx**. If you wish to extract other data evaluated in the simulation, you can do so by making slight modifications to **B_Smooth.py**.
