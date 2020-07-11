echo "-------------------------------------"
echo "cmd in_file out_file"
echo "-------------------------------------"
echo 
echo 
echo 

ffmpeg -i $1 -q:a 0 -r 30 -strict -2 $2
