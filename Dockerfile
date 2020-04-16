FROM pytorch/pytorch:latest

# copy requirements descriptor
COPY ./requirements.txt /requirements.txt

# install dependencies
RUN pip install --upgrade pip
RUn pip install --upgrade pipenv
RUN pip install --upgrade -r /requirements.txt

# copy remaining files
COPY . .

# Install Node.js
RUN curl -sL https://deb.nodesource.com/setup_8.x | bash
RUN apt-get install --yes nodejs


# install node dependencies and build web ui
RUN npm install -g @vue/cli @vue/cli-service-global
RUN cd public/public/ && npm install && cd src/ && vue build

# start server
ENTRYPOINT [ "python" ]
CMD [ "server.py" ]