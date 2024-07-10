#!/bin/bash

printline() {
	echo "==========================================================="
}

printline

echo -n "Checking Python 3.10.12 or higher... "

VERSION=$(python3 -V | awk '{print $2}')
REQUIRED_VERSION="3.10.12"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "ERROR."
    echo "    (1) You need Python 3.10.12 or higher to run MH-FSF!"
    echo "    (2) Please, install Python 3.10.12 or higher, or use the Docker demo (run_demo_docker.sh)."
    printline
    exit 1
fi

echo "done."
printline

printline
echo -n "Installing Python Requirements ... "

pip install -r requirements.txt  > /dev/null 2>&1

echo "done."
printline

echo ""

printline
echo "Running MH-FSF ... "
echo ""

python3 main.py run --all-fs-types --fs-methods pca lr mt anova semidroid -d datasets/example.csv --all-ml-models --parallelize

echo ""
echo "done."
printline
