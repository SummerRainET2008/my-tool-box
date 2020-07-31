sudo apt-get update
sudo apt-get remove docker docker-engine docker.io
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update

export version=5:19.03.12~3-0~ubuntu-bionic;   
sudo apt-get install docker-ce=$version docker-ce-cli=$version containerd.io
#sudo apt-get install docker-ce docker-ce-cli containerd.io

