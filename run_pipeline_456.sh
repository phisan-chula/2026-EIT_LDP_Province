#!/bin/bash

# Check if the HASC_1 argument is provided
if [ -z "$1" ]; then
  echo "Error: Missing HASC_1 argument."
  echo "Usage: $0 <HASC_1>"
  echo "Example: $0 TH.AC"
  exit 1
fi

# Assign the first argument to a variable for clarity
HASC_1="$1"

echo "Starting processing for $HASC_1..."

# Run the Python scripts sequentially
python 4_analyse_ldp.py "$HASC_1"
python 5_plot_PP_PctPopu.py "$HASC_1"
python 6_LDP_OnePage.py --hasc "$HASC_1"

echo "Finished processing $HASC_1!"
