cat $1 | awk 'BEGIN { sum = 0; } { sum += NF; } END { print sum/NR}'
