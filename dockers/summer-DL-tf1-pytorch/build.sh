echo "FROM tensorflow/tensorflow:1.15.0-gpu-py3"  > Dockerfile
cat ../Dockerfile.share >> Dockerfile

sudo docker build -t summer-dl-tf1-pytorch .
rm Dockerfile
