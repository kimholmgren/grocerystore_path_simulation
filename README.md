# Grocery Store Customer Path Simulation
Before a grocery store opens key operational decisions must be made 
with no historical data. One important decision is how to optimally 
lay out the store to maximize consumer spending. This work reviews 
existing literature on simulation to optimize grocery store layout, 
uses computer vision techniques to transform a store diagram into a 
digital representation, and applies simulation methods to approximate
 which of the layouts proposed by a store designer would result in the
  highest amount of impulse purchasing. Output analysis methods are 
  used to compare these results to determine whether one design 
  outperforms the others. 
  
## Installation
Navigate to this repository in the command line. Use   
`pip install .`  
to install the grocerypathsim package. Installing jupyter is recommended as well in order to view the Usage notebook.  
Additionally, tesseract must be installed for OCR.   
**Mac**  
`brew install tesseract`  
  
  **Linux**  
  `sudo apt update`  
  `sudo apt install tesseract-ocr`    
  `sudo apt install libtesseract-dev`
  

## Getting Started
See Usage.ipynb for examples and directions on how to use this package.  
In order to use the 2017 Instacart Grocery Shopping Dataset the necessary files must first be downloaded and placed in the same directory with the filenames "departments.csv",
"products.csv", and "order_products__train.csv". Otherwise, create a different dataset and generator for grocery lists in shopping_lists.py.