import os
from pathlib import Path

def remove_wrapper_head(folder):
    for f in os.listdir(folder):
        name=Path(f).parts[-1]
        file="../benchmarks_sources/{}/{}_generator.cpp".format(name, name)
        write_str=""
        with open(file, 'r') as f:
            original_str = f.readlines()
        for l in original_str:
            if "wrapper.h" in l:
                l=""
            write_str+=l
        f.close()
        with open(file, 'w') as f:
            f.write(write_str)

if __name__ == '__main__':
    remove_wrapper_head("../benchmarks_sources") 