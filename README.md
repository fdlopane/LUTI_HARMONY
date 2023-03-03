# HARMONY-Components-Python
This repository is for gathering the code of the different HARMONY components that are implemented in Python.

To run the project, simply issue:
```
sh deploy.sh
```
We assume that you use Python 3.7!

## Working with git and Github

1. Create a branch for your component. This can be done by navigating to the
[HARMONY-Components-Python project on Github](https://github.com/MobyX-HARMONY/HARMONY-Components-Python), 
clicking on the `main` button below the `Issues` button on the top left, typing a name for your branch 
(e.g. `eem`) and clicking on **Create branch: eem** below the name.  
2. Clone the project, go to the cloned folder, and switch to your branch (called e.g. `eem`).
    In command line, you can do the above by:
    ```
    git clone https://github.com/MobyX-HARMONY/HARMONY-Components-Python.git
    cd HARMONY-Components-Python
    git checkout eem
    ```
3. Once you have done your changes (see below on what to change), you need to commit the changes and push them to Github. 
You can do the above in the command line by:
   ```
   git commit . -m "add here a commit message that describes your changes"
   git push
    ```
   That's it!

## What to change

1. Add your code in the `src` directory. 
2. Create a directory `outputs` under `src`, if it doesn't already exist.
3. Update file `main.py` to start your component. 
You can remove everything and only these two lines:
   ```
    if __name__ == '__main__':
    from config import inputs, outputs
    ```
4. Add your dependencies in `requirements.txt`: Add one line for each package, then a version number 
following the example in the first line (`python-schema-registry-client==1.8.1`). 
For more information check "Requirements Files" on the [Pip User Guide](https://pip.pypa.io/en/stable/user_guide/).
5. Update `config.py` with the inputs and outputs of your component.  
6. Add your input files in the directory `inputs`.
7. Update `component.json` with the description of your component: name, description, inputs, and outputs.
    - each input can be of type `textbox`, `dropdown`, or `file`
      - if an input is of type `textbox`, then you need to provide also a `scalarType` which can be one of
      `string`, `int` or `float`
    - all outputs are of type `file`
    - `label` is the name what will be shown in the user interface of HARMONY MS
    - `description` will be shown in the user interface of HARMONY MS as well
    - `key` should match the key in the `config.py`
    - `required` should be always present and set to `true` for each input
    - `value` for an input is either the default value (for inputs of type `textbox`)
    or should be present and set to the empty string (for inputs of type `file`)
    - `value` for outputs should be the name of the file, without the full path. 
