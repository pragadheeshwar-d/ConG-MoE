import os
import sys

# Ensure src directory is in python search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict_interactive import main

if __name__ == "__main__":
    main()
