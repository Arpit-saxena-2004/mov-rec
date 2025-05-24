
#!/bin/bash

# Install PyTorch CPU version from the official PyTorch wheel index
pip install torch==2.7.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html


# Then install other dependencies
pip install -r requirements.txt
