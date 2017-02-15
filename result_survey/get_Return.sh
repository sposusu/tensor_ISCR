cat $1 | grep INFO | grep MAP | cut -f 2 | cut -d' ' -f3 
