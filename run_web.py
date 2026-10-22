import os
import sys

# Ensure src directory is in python search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from web_app import run

if __name__ == "__main__":
    run()
