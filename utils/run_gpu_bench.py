#!/usr/bin/env python3

import subprocess
import csv
import sys

result = subprocess.run(["nvprof", "--print-gpu-trace", "--csv", sys.argv[1]], stderr=subprocess.PIPE, stdout=subprocess.PIPE)


lines = result.stdout.split(b'\n')
# print(b'\n\n---\n\n'.join(lines).decode())

for i, l in enumerate(lines):
    if l.startswith(b','):
        tiramisu_exec_times = [float(num.decode()) for num in lines[i].split(b',')[1:]]
        halide_exec_times = [float(num.decode()) for num in lines[i+1].split(b',')[1:]]
        lines = lines[:i] + lines[i + 2:]
        break

print(b'\n'.join(lines).decode())

lines = result.stderr.split(b'\n') 


for i, line in enumerate(lines):
    if line.endswith(b'Profiling result:'):
        start_index = i + 1
        break

reader = csv.DictReader(l.decode() for l in lines[start_index: -1])
rows = [row for row in reader]
units = rows[0]
rows = rows[1:]

nb_tests = len(tiramisu_exec_times) + 1

assert(len(rows) % nb_tests == 0)

def check_equal(rows, start1, start2, length):
    for i in range(length):
        if rows[start1 + i]["Name"] != rows[start2 + i]["Name"]:
            return False
    return True

for i in range(1, len(rows)//nb_tests):
    a_len = i
    b_len = len(rows)//nb_tests - a_len
    # heuristic check
    if (all(check_equal(rows, 0,     j        * a_len +     b_len, a_len) for j in range(1, nb_tests)) and
        all(check_equal(rows, a_len, nb_tests * a_len + j * b_len, b_len) for j in range(1, nb_tests))):
        break

def median(exec_times, rows, start, length):
    sorted_indices = sorted(list(range(len(exec_times))), key=lambda i: exec_times[i])

    # print(sorted_indices)
    
    nb = len(exec_times)
    result = []
    first, second = sorted_indices[(nb - 1) // 2], sorted_indices[nb // 2]
    result.append(('Exec Time', (exec_times[first] + exec_times[second]) / 2))

    for (row1, row2) in zip(rows[start + length * first: start + length * (first + 1)], rows[start + length * second: start + length * (second + 1)]):
        result.append((row1['Name'], (float(row1['Duration']) + float(row2['Duration'])) / 2))
    total_copy = 0.0
    total_kernel = 0.0
    for label, number in result[1:]:
        if label.startswith('[CUDA memcpy'):
            total_copy += number
        else:
            total_kernel += number
    result.append(('Total Copy', total_copy))
    result.append(('Total Kernel', total_kernel))
    return result

def print_result(result):
    for label, number in result:
        print("{:40} {}".format(label[:37], number))

# print([row['Name'] for row in rows[:a_len]])
# print([row['Name'] for row in rows[a_len: a_len + b_len]])

print('== tiramisu ==')
print_result(median(tiramisu_exec_times, rows, a_len + b_len, a_len))
print('== halide ==')
print_result(median(halide_exec_times, rows, nb_tests * a_len + b_len, b_len))

# print(len(rows))
# print(rows[0])
# print(rows[-1])

