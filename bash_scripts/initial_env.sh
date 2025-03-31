#!/bin/bash

# purge all modules
module purge
if [ $? -eq 0 ]; then
	echo "Module purge : SUCCESS"
else
	echo "Module purge : FAILED"
	exit 1
fi

# spartan
module load spartan
if [ $? -eq 0 ]; then
    echo "Module spartan: SUCCESS"
else
    echo "Module spartan: FAILED"
    exit 1
fi

# foss/2022a
module load foss/2022a
if [ $? -eq 0 ]; then
    echo "Module foss/2022a: SUCCESS"
else
    echo "Module foss/2022a: FAILED"
    exit 1
fi

# Python/3.10.4
module load Python/3.10.4
if [ $? -eq 0 ]; then
    echo "Module Python/3.10.4: SUCCESS"
else
    echo "Module Python/3.10.4: FAILED"
    exit 1
fi

# SciPy-bundle/2022.05
module load SciPy-bundle/2022.05
if [ $? -eq 0 ]; then
    echo "Module SciPy-bundle/2022.05: SUCCESS"
else
    echo "Module SciPy-bundle/2022.05: FAILED"
    exit 1
fi

echo "All Done !! "
