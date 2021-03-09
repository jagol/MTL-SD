for dir in data/*; do
  for file in $dir/{train,dev,test}.jsonl; do
    echo $file
    shuf $file -o $file
  done
done