#!/usr/bin/env bash

# Runs a command in parallel across all nodes
# Usage
# sync_cmd.sh 'echo "hello world"'

echo "Command: $1";
pdsh -R ssh -w ^/job/hosts $1
