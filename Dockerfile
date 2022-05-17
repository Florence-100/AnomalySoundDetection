#Set base image in Python 3.9
FROM python:3.9

#Expose Port 8501 for app to be run on 
EXPOSE 8501

#Set Working Directory 
WORKDIR /app

#copy packages required from local requirements file to Docker image requirements file 
COPY requirements.txt ./requirements.txt

#Run command line instructions 
RUN pip3 install -r requirements.txt

#Copy all files from local project folder to Docker image 
COPY . . 

#Command to run Streamlit application
CMD streamlit run simple_streamlit_application.py