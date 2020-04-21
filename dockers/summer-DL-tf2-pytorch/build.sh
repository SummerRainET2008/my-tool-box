echo "FROM tensorflow/tensorflow:nightly-gpu-py3"  > Dockerfile
cat ../Dockerfile.share >> Dockerfile
echo "RUN python3 -m pip install tensorflow" >> Dockerfile

sudo docker build -t summer-dl-tf2-pytorch .
rm Dockerfile
