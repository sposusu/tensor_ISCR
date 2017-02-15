cat $1 | grep INFO | grep MAP | cut -f 1 | cut -d' ' -f 3
