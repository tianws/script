#!/usr/bin/env bash

ls *.jpg > filepath.txt
filepath=$(cd "$(dirname "$0")";pwd)
echo "$(basename $0) $(dirname $0) --$filepath"
#sed -i "s/^/$filepath\//g" filepath.txt
filepath=$(echo $filepath | sed 's:\/:\\/:g')
echo $filepath
echo $filepath | xargs -I {} sed -i 's:^:{}\/:g' filepath.txt
