feat=$1
net=$2
log_dir="./log/$feat""_""$net"
model_dir="./output/orca_binary/$feat""_""$net"
rm $log_dir
rm -r $model_dir
echo "deletion finished"