FROM tensorflow/tensorflow:nightly-gpu-py3
 
# Update Software repository
RUN apt-get update
 
# Install nginx, php-fpm and supervisord from ubuntu repository
RUN apt-get -y install vim 
RUN python3 -m pip install torch 
RUN python3 -m pip install scipy 
RUN python3 -m pip install sentencepiece 
RUN python3 -m pip install nltk
RUN python3 -m pip install params_flow
RUN python3 -m pip install flask 
RUN python3 -m pip install pillow 
RUN python3 -m pip install allennlp 
RUN python3 -m pip install word2number
RUN python3 -m pip install sklearn 
RUN python3 -m pip install Pillow 
RUN python3 -m pip install matplotlib 
RUN python3 -m pip install beautifulsoup4 
RUN python3 -m pip install html5lib 
RUN python3 -m pip install httplib2 
RUN python3 -m pip install jieba 
RUN python3 -m pip install transformers 
RUN python3 -m pip install torchvision 
RUN python3 -m pip install rasterio
RUN python3 -m pip install torch_lr_finder
 
ENV PYTHONPATH /root/summer201/my-tool-box/insight_nlp:$PYTHONPATH 

# write full path!
COPY bashrc /root/.bashrc
COPY local_bashrc /root/.local_bashrc

