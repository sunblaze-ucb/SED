from sys import argv

import json

first, second, output = argv[1:]

with open(first) as f:
    fc = json.load(f)

with open(second) as s:
    sc = json.load(s)


result = [a + b for a, b in zip(fc, sc)]

with open(output, "w") as o:
    json.dump(result, o)