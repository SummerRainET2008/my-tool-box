FROM tensorflow/tensorflow:1.15.0-gpu-py3
 
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
 
#Define the ENV variable
#ENV nginx_vhost /etc/nginx/sites-available/default
ENV PYTHONPATH /summer201/my-tool-box/insight_nlp:$PYTHONPATH 

# Volume configuration
#VOLUME ["/home/summer/summer201"]
 
# Configure Services and Port
#COPY start.sh /start.sh
#CMD ["./start.sh"]
COPY bashrc /root/.bashrc
 
#EXPOSE 80 443
