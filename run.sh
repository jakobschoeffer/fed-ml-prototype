#!/bin/bash

python server.py &
sleep 2 # Sleep for 2s to give the server enough time to start
python client1.py &
python client2.py &
python client3.py &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
