# Overview
This code is designed to predict fatigue life and limit of steels based on microstructure information, tensile properties, and loading conditions.

# Requirements
* [**Abaqus**]: This is required to generate .dat file.
  * .
  * .
 
* [**Python 3.10**](https://www.python.org/downloads/): This is necessary for running the simulation.
  * Additionally, the following libraries are required: os, re, math, Random, pickle, numpy, pandas, datetime and scipy. You can install these libraries using the following commands:
    `pip install os, re, math, Random, pickle, numpy, pandas, datetime, scipy`

# How to Execute
1. Generate the **polycrystal.tess** file using the Neper Tessellation Module.
   * The **polycrystal.tess** file includes all the data describing the polycrystalline structure.
   * We have provided a sample file for your convenience.
2. Modify the values in the **parameters.dat** file.
   * The **parameters.dat** file includes all the required input parameters other than the data describing the polycrystalline structure, such as the target temperature, applied stress tensor, material constants, and numerical conditions.
3. Place **polycrystal.tess**, **parameters.dat**, and **creep.py** in the same directory, and execute **creep.py**. The strain and void area fraction data will be exported in the output file named **results.dat**. If you wish to extract other data evaluated in the simulation, you can do so by making slight modifications to **creep.py**.
