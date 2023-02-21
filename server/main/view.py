#import module
from flask import Flask, url_for, render_template, redirect, request
import os
from main import app

#load environment variables into module
from dotenv import load_dotenv
load_dotenv()

#get secret key from environment
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

@app.route('/')
def index():
	return render_template ('index.html')