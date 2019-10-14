# pythonTFM
Traction Force module using python (FOSS)

## Motivation
 - Attempt to adapt Matlab TFM to python - free to use.
 - Using jupyter for beter visualization of the process and web friendly.
***For overview of traction force microscopy method please visit my matlab TFM repo: https://github.com/htmtri/matlabTFM/

## Status
Pre-processing: 
 - Images input/cropping: complete
 - Cell tracing: complete
PIV (only cross-ocorrelation algorithm avaiable; MQD algorithm unavaiable atm):
 - Pre-process filtering: complete 
 - Stage dedrift: complete
 - PIV: complete
 - Post-process: 
    - bogus: complete
    - noise: to be done
    
Solving for force: to be done (not soon)
Result from PIV can readily be used with ANSYS to solve for traction force similar to matlab TFM. However, the purpose of this project is to make everything free, another approach is needed. (all finite element solving requires very heavy computation and optimization and thus most if not all finite element softwares' license cost a ton). Unfornately, solving Boussinesq solution to Green's function in Fourier space (FTTC) is also computationally intensive and hence the need to use C/C++ emerge. 

# Usage
