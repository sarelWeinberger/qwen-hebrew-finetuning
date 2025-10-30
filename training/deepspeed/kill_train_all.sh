#!/bin/bash
ps ux | grep 'train.py' | awk '{print $2}' | xargs kill -9
ps ux | grep 'torch/_inductor/compile_worker/__main__.py' | awk '{print $2}' | xargs kill -9
