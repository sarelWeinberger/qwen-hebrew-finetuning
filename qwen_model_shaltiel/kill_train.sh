#!/bin/bash
# Kill all python processes running train.py
ps aux | grep '[p]ython.*train.py' | awk '{print $2}' | xargs -r kill -9

echo "All train.py processes have been killed."
