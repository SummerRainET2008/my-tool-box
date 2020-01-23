echo "FROM tensorflow/tensorflow:nightly-gpu-py3"  > Dockerfile
cat ../Dockerfile.share >> Dockerfile

sudo docker build -t summer-dl-tf2-pytorch .
rm Dockerfile
