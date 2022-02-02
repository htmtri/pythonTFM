# pythonTFM
Traction Force module using python (FOSS)

## Motivation
- Attempt to adapt Matlab TFM to python - free to use.
- Using jupyter for beter visualization of the process and web friendly.
  ***For overview of traction force microscopy method please visit my matlab TFM repo: https://github.com/htmtri/matlabTFM/

## Status
Pre-processing:
- Images input/cropping: completed
- Cell tracing: completed
  PIV (only cross-ocorrelation algorithm avaiable; MQD algorithm unavaiable atm):
- Pre-process filtering: completed
- Stage dedrift: completed
- PIV: completed
- Post-process:
    - bogus: completed
    - noise: completed

Solving for force: to be done (not soon)
Result from PIV can readily be used with ANSYS to solve for traction force similar to matlab TFM. However, the purpose of this project is to make everything free, another approach is needed. (all finite element solving requires very heavy computation and optimization and thus most if not all finite element softwares' license cost a ton). Unfornately, solving Boussinesq solution to Green's function in Fourier space (FTTC) is also computationally intensive and hence the need to use C/C++ and Cython.

# Usage


# Credit
- Qi Wen, the original author of static matlab TFM code https://www.wpi.edu/people/faculty/qwen
- Ujash Joshi, University of Toronto for normalized cross-correlation code. https://github.com/Sabrewarrior/normxcorr2-python/master/norxcorr2.py
- Daniel Blair and Eric Dufresne for implementation of PT_IDL code with image filter in Matlab http://site.physics.georgetown.edu/matlab/
- Jörg Döpfert, Flix.TECH. https://github.com/jdoepfert/roipoly.py
- All authors and contributors of numpy, matplotlib, scipy, opencv, openpiv:
    - https://numpy.org
    - https://matplotlib.org
    - https://www.scipy.org
    - https://opencv.org
    - www.openpiv.net
