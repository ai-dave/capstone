# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY app.py /app
COPY requirements.txt /app
#COPY model-human.pkl /app
#COPY CountVectorizer-human.pkl /app
COPY templates/ /app/templates
COPY static/ /app/static
COPY model/ /app/model
COPY bioinformatics/ /app/bioinformatics

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
EXPOSE 8000
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]
CMD ["app:app"]
