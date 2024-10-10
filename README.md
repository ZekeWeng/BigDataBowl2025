# BigDataBowl2025



## Instructions for Start

```
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Prepare data file
python preprocessing.py
```


### Timeline:
October 11: Challenge Opens \
October 25: Plan Made \
December 1: Rough Draft Done \
January 6: Challenge Ends


### Todos:
Visualizations
 - Frankenstein Viz
 - Toggles
    - See Numbers
    - Offense Only
    - Defense Only
    - Vertical (rotation of the field cuz currently horizontal)
 - Change The Color
 - Increase the field width to look better


Creating VENV

deactivate  # First, deactivate the virtual environment
rm -rf venv  # Be cautious with rm -rf
python3 -m venv venv  # Create a new virtual environment
source venv/bin/activate  # Activate the new environment



pip freeze > requirements.txt

pip uninstall package_name