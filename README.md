# Master-s_thesis
Selected code and files from my Master's thesis, 
where i created and evaluated a predective VAR model 
that accelerated molecular dynamics simulations.

------ Master_thesis_Final.pdf ------ 
PDF file of my written master's thesis.


------ Master_thesis_outline.pdf ------ 
PDF file of the outline of my thesis.


------ Oral_defense.ppt ------ 
PPT that i used to defend my thesis.


------ Meetings/ ------ 
Folder with powerpoints that I used to present my work 
at weekly meetings


------ Data/ ------ 
Folder with selected data from testing model


------ SVD_model.py ------ 
Python file with the code for vector auto-regressive model.


------ Data_analysis.ipynb ------
Jupyter notebook with the data analysis evaluating the model.


------ RNN_framework.ipynb ------ 
Jupyter notebook with code for a RNN model that forecasts molecular forces


------ Simulation_with_SVD_model.py ------ 
Py file that runs a molecular simulation using ASE and GPAW
where the forces in some steps are predicted by the model
instead of calculated with GPAW.


------ Model_integration_gpaw.ipynb ------ 
Jupyter notebook showing how I integrated the model with GPAW and
made sure the implementation was correct.