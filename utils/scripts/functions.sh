#!/bin/bash

echo_and_run_cmd () { # takes one argument, the command
    echo $1
    # Run the command
    $@
}
