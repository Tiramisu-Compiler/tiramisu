#!/bin/bash

find . -name Makefile  | xargs dirname | xargs -I NAME make -C NAME clean
find . -name Makefile  | xargs dirname | xargs -I NAME make -C NAME
