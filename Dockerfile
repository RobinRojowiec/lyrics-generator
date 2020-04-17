FROM anibali/pytorch:no-cuda

# copy requirements descriptor
COPY ./requirements.txt /requirements.txt

# install dependencies
RUN pip install --upgrade -r /requirements.txt

# copy remaining files
COPY . .

# Install Node.js
RUN sudo apt-get update --yes
RUN sudo apt install --yes build-essential apt-transport-https lsb-release ca-certificates curl
RUN sudo curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
RUN sudo apt-get install --yes nodejs

# install node dependencies and build web ui
RUN sudo npm install -g @vue/cli @vue/cli-service-global
RUN cd frontend/ && sudo npm install && sudo npm run build

# start server
ENTRYPOINT [ "python" ]
CMD [ "server.py" ]