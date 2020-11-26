from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
#Login to Google Drive and create drive object
g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)
# Importing os and glob to find all PDFs inside subfolder
from glob import glob
from tqdm import tqdm
import os
files = glob.('C:/train_image2/*')
for i in tqdm(range(len(files))):
    with open(files[i],"r") as f:
        fn = os.path.basename(f.name)
        file_drive = drive.CreateFile({'title': fn })  
    file_drive.SetContentString(f.read()) 
    file_drive.Upload()
print "All files have been uploaded"