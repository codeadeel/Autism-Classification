#!/usr/bin/env python3

# Importing Libraries
import argparse
import string
import base64
from io import BytesIO
from PIL import Image
import requests

# %%
# Inference Tools
class Classify:
    def __init__(self, url: string) -> None:
        """
        This class is used to initialize classification client

        Method Input
        =============
        url : Inference Server URL

        Method Output
        ==============
        None
        """
        self.server_url = url
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def __encodeImage__(self, addr) -> string:
        """
        This method is used to encode image for inference

        Method Input
        =============
        addr : Absolute address of image

        Method Output
        ==============
        Encoded Image
        """
        img = Image.open(addr)
        byt1 = BytesIO()
        img.save(byt1, format = 'JPEG')
        bdat = byt1.getvalue()
        return base64.b64encode(bdat).decode('utf-8')
    
    def __call__(self, data) -> tuple:
        """
        This method is used to handle image's inference from inference server
        
        Method Input
        =============
        data : Absolute address of image's in the form on single string or list

        Method Output
        ==============
        Infered data & response status code
        """
        if type(data) == str:
            edat = self.__encodeImage__(data)
            resp = requests.post(self.server_url, headers = self.headers, json = [edat])
        elif type(data) == list:
            edat = [self.__encodeImage__(i) for i in data]
            resp = requests.post(self.server_url, headers = self.headers, json = edat)
        else:
            raise Exception('Only Sring or List Address Formats are Allowed')
        return resp.json(), resp.status_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Autism Classification Inference Client.')
    parser.add_argument('-l', '--links', nargs = '+', help = 'Absolute Address of Subject Images', required = True)
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to REST Server => IP:Port', required = True)
    args = vars(parser.parse_args())
    cla = Classify(args['server_ip'])
    res, scode = cla(args['links'])
    print('=' * 30)
    print(f'Inference Results: {res}')
    print(f'Status Code: {scode}')
    