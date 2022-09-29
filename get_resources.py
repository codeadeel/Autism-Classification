#!/usr/bin/env python3

# %%
# NOTE
# ====
# This file is responsible to download trained resources from Google Drive. Please run this script after cloning this repository to get trained resources.
#
# To install gdown, please visit following repository.
#		https://github.com/wkentaro/gdown

# %%
# Importing Libraries
import os
import gdown

# %%
# Download Configuration

dlinks = {
    'resnet50-11ad3fa6.pth': 'https://drive.google.com/file/d/1-uKTPTMHY8aXKmwqJSb7UEwI2Xlpek5l/view?usp=sharing',
    '/pretrained/labels': 'https://drive.google.com/file/d/1UoIW65813fgDZP7VGGHA7rNLK8_iDj_Z/view?usp=sharing',
    '/pretrained/logs/logs': 'https://drive.google.com/file/d/11WDxhUow63NwOcI2175iUPIpBQI-VjTU/view?usp=sharing',
    '/pretrained/models/model_21': 'https://drive.google.com/file/d/1wwt0-3aq0uiddJp4GjKd83QpnWx6en-y/view?usp=sharing',
    '/pretrained/models/model_24': 'https://drive.google.com/file/d/1NaJdDXTTS_SQGFcRtVMHe3GPDDEim1x0/view?usp=sharing'
}

# %%
# Download Execution

download_addr = '/'.join(__file__.split('/')[:-1]) + '/Resources'
os.mkdir(download_addr)
os.mkdir(f'{download_addr}/pretrained')
os.mkdir(f'{download_addr}/pretrained/logs')
os.mkdir(f'{download_addr}/pretrained/models')

for keys, vals in dlinks.items():
    gdown.download(vals, f'{download_addr}/{keys}', quiet=False, fuzzy=True)
    