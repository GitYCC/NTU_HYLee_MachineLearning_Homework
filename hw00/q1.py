#!python2.7
import sys

index = int(sys.argv[1])
input_file = str(sys.argv[2])


def convert_line_to_nums(line):
    words = line.split(' ')
    nums = map(float, words)
    return nums


col_nums = list()
with open(input_file) as f:
    for line in f.readlines():
        line = line.strip()
        if line != '':
            nums = convert_line_to_nums(line)
            col_nums.append(nums[index])

sorted_col_nums = sorted(col_nums)
output = ','.join(map(str, sorted_col_nums))
print output

open('result/ans1.txt', 'w').write(output)
