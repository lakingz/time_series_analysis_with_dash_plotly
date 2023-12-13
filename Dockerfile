#
# python
#

# Start your image with a node base image
FROM python:3.8

# The /app directory should act as the main application directory
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy local directories to the current local directory of our docker image (/app)
COPY ./app.py /app/app.py
COPY ./PJME_hourly.csv /app/PJME_hourly.csv

EXPOSE 8050
# Start the app using serve command
CMD ["python", "app.py"]