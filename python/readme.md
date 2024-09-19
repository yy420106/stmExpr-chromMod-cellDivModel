# README

## Description

The python scripts in the same folder of this file, provided a CLI style program for simulation and visualization of ***STM* expression-chromatin modification-cell division coupled model**, which was utilized and described in the following research paper:

A cell cycle-dependent epigenetic Sisyphus mechanism maintains stem cell fate for shoot branching (DOI: XXX)

For detailed explanation of the model, please see the **Supplement_Computational_Methods_and_Simulation_Details.pdf**.

This program contains 6 base subcommands, each of which performs a specific task:

- ***schmdg***:
    plot schematic diagram to explain model mathematical principles
- ***epistb***:
    plot the profiles of methylation level and *STM* expression change over multiple days in both stem cells and 
    differentiated cells to show epigenetic stability
- ***dynmcyc***:
    plot the influence of dynamic (increasing/decreasing) cell cycle on stem cells
- ***divarrest***:
    plot transition curves of stem cell differentiation during different length of division arrest & cell type 
    distribution statistics after division recovery
- ***rescue***:
    plot recovery curves of differentiated cells after divison arrest using different rescue stategies
- ***bimap***:
    plot bistability heatmap in 2-D parameter $k_{me}$ - $P_{dem}$ space under different cell cycle conditions
    
The main intergace of program is `main.py`, use "-h/--help" option to get the help/usage informations of this script and each subcommand. Before using, make sure that your current python environment contains the necessary dependency libraries, which are listed in ***requirements.txt***.


**Author**: Yi Yang, School of Life Science, PKU, Beijing, China. (contact: 2301110575@pku.edu.cn)

**Version**: 0.0.1

## Example Usage

    python main.py -h
    python main.py schmdg -h
    python main.py divarrest -h
    
    python main.py schmdg -d
    python main.py epistb -ns 80 -et 0 3 -md 40 -ts 10 -d
    python main.py epistb -ns 80 -et 0 3 -md 40 -c -d -t
    python main.py dynmcyc -ns 80 -df 0.5 3 -ec 12 -md 30 -d
    python main.py divarrest -ns 800 -ec 12 -md 35 -ad 1 3 5 7 9 11 13 15 17 19 -ps 10 -cd 8 16 -d
    python main.py divarrest -ns 80 -ec 12 -md 35 -ad 1 3 5 7 9 11 13 15 17 19 -a -c -d -t
    python main.py rescue -ns 80 -rs M A S -ad 20 -td 25 -rd 5 -d
    python main.py bimap -ns 100 -cc 11.0 22.0 44.0 88.0 -nc 40 -mp 101 -d
