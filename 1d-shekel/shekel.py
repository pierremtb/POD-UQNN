import sys

eqnPath="1d-shekel"
sys.path.append(eqnPath)
from pod import get_pod_bases

# Getting the POD bases
x, V = get_pod_bases()

