#!/usr/local/bin/python2.7
import sys

index = int(sys.argv[1])
input_file = str(sys.argv[2])

with open(input_file) as f:
    line = f.readlines()[index].strip()
    words = line.split(' ')
    nums = map(float, words)
    sorted_nums = sorted(nums)
    output = ','.join(map(str, sorted_nums))
    print output

open('ans1.txt', 'w').write(output)
