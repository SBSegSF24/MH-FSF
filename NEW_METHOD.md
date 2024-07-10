## How to add new methods

To add new features selection methods to tool, follow steps below:

 - create a new directory with method identifier in **methods** directory, within the directory of the intended type.
  - In this new directory, add two files: **about.desc**, containing the method description, and **run.py** (template [here](template_method.py)), based on a standard structure.
   - In **run.py**, import the necessary libraries and define two functions:
	   - **add_arguments**, which adds specific arguments to argparse.ArgumentParser;
	   -  and **run**, which performs feature selection and saves the reduced dataset. In the **run** function, use the same directory name as method_id to save reduced dataset to path specified by the arguments.
