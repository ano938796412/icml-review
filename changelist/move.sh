#!/bin/bash

# move .py in directory to corresponding locations in `locations.txt`

input="locations.txt"
while read -r line; do
  echo "$line"
  cp "$(basename "$line")" "$(dirname "$line")"
done <"$input"
