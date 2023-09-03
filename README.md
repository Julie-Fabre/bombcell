# 🪄 prettify matlab
My one-stop shop to instantly make your MATLAB scripts and plots beautiful, publication-ready and colorblind friendly.

## :triangular_flag_on_post: Features 

### Prettify plots 

Run `prettify_plot;` to prettify your current figure (includes all subplots). Includes options to modify the background color, text size and homogenize x and y limits across plots. 
<img src="./images/prettify_plot.png" width="100%">

### Prettify code

Run `prettify_current_code;` to prettify your current script open in the MATLAB editor. Prettify rules are stored in the `formatRules.xml` file. 
<img src="./images/prettify_code.png" width="100%">

### Prettify colors
#### Colorblind simulator 
Run `prettify_colorblind_simulator;` to plot your current figure as it would seen with different types of color blindness.

#### Colormaps
[Coming soon] Perceptually-uniform, colorblind friendly colormaps. 


## 🏁 Installation

To use prettify_matlab:

- clone the repository
- add the repository folder to MATLAB's path.

prettify_matlab doesn't use any of MATLAB's add-on toolboxes and doesn't have any dependancies. 



---
To do:
- integrate my colorblind simulator
- perceptually-uniform colormaps 
- inkscape / scientific 
