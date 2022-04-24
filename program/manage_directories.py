import os
import shutil

def delete_directory(path):
    try:
        shutil.rmtree(path) #delete directory with files
    except OSError:
        print ("Deletion of the directory %s failed" % path)
    else:
        create_directory(path)

def create_directory(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        
def create_results_directory():
    current_path = os.getcwd()
    
    path = current_path + '/results'
    
    is_exists = os.path.isdir(path)
    if is_exists :
        delete_directory(path)
    else: 
        create_directory(path)

def manage_directories(directory_name):
    current_path = os.getcwd()
    
    path = current_path + '/results/' + directory_name
    is_exists = os.path.isdir(path)
    if is_exists :
        delete_directory(path)
    else: 
        create_directory(path)