#!/usr/bin/env python3

# %%
# Importing Libraries
from model import *
import argparse
import string
import math
import base64
from io import BytesIO
import tqdm
from flask import Flask, request, jsonify

# %%
# Main Data Loading Class
class Data(torch.utils.data.Dataset):
    def __init__(self, datAddr: string, transform = inference_transforms) -> None:
        """
        This method is used to initialize data loading class
        
        Method Input
        =============
        datAddr : Absolute address of data directory for clustering
                            Default Directory Hierarchy:

                                data
                                  |_ class-1
                                  |       |_ sample-1
                                  |       |_ sample-2
                                  |       |_ ...
                                  |
                                  |_ class-2
                                  |       |_ sample-1
                                  |       |_ sample-2
                                  |       |_ ...
                                  |_ ...
        
        transform : Subject transforms to apply on the data
        
        Method Output
        ==============
        None
        """
        super(Data, self).__init__()
        self.dataset_address = datAddr
        self.transform = transform
        self.classes = {j:i for i, j in enumerate(os.listdir(self.dataset_address))}
        self.rev_classes = {j:i for i, j in self.classes.items()}
        self.filenames = list()
        self.filelabels = list()
        for i in list(self.classes.keys()):
            for j in os.listdir(f'{self.dataset_address}/{i}'):
                self.filenames.append(f'{self.dataset_address}/{i}/{j}')
                self.filelabels.append(self.classes[i])
    
    def __len__(self):
        """
        This method is used to find the number of files in respective dataset
        
        Method Input
        =============
        None
        
        Method Output
        ==============
        None
        """
        return len(self.filenames)
    
    def __getitem__(self, idx) -> tuple:
        """
        This method is used to load and process the image based on file number
        
        Method Input
        =============
        idx : File number ( 0 -> self.__len__())
        
        Method Output
        ==============
        Processed Image Data, Respective Image One Hot Encoded Label
        """
        img = Image.open(self.filenames[idx])
        timg = self.transform(img)
        return timg, torch.Tensor([self.filelabels[idx]]).type(torch.int32)

# %%
# Main Inference Class
class Inference:
    def __init__(self, resourceAddr: string, ethres: float = 0.75, batch_size: int = 32, datAddr:string = '/data') -> None:
        """
        This method is used to initialize inference class for Autism Classification

        Method Input
        =============
        resourceAddr : Absolute address of data directory with trained resources
        ethres : Similarity threshold for one shot sample matching for Clustering Mode ( default : 0.75 )
        batch_size : Batch size for computing base embeddings for Clustering Mode ( default : 32 )
        datAddr : Absolute address of base data directory for Clustering Mode

        Model Output
        =============
        None
        """
        self.resources_address = resourceAddr
        self.embedding_thres = ethres
        self.batch_size = batch_size
        self.clustering_data = datAddr
        if len(os.listdir(self.clustering_data)) == 0:
            self.embedding_mode = False
        else:
            self.embedding_mode = True
        self.__device__ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('%' * 10 + ' Warming Up ' + '%' * 10)
        with open(f'{self.resources_address}/labels', 'rb') as file1:
            self.labels = pickle.load(file1)
            self.rev_labels = {j:i for i, j in self.labels.items()}
        self.__modelChooser__()
        torch.cuda.empty_cache()
        self.mod = Model(len(self.labels))
        if self.embedding_mode == True:
            self.mod.load_state_dict(torch.load(f'{self.resources_address}/models/{self.embedModel}', map_location = self.__device__))
        else:
            self.mod.load_state_dict(torch.load(f'{self.resources_address}/models/{self.classModel}', map_location = self.__device__))
        self.mod.to(self.__device__)
        self.mod.eval()
        if self.embedding_mode == True:
            self.__createCluster__()
    
    def __str__(self) -> string:
        """
        Method to display metadata about subject class

        Method Input
        =============
        None

        Method Output
        ==============
        Returns new line after printing subject metadata
        """
        print('=' * 30)
        if self.embedding_mode == True:
            labs = list(self.embedding_classes.values())
            print('Inference Mode: Clustering')
            print(f'Labels: {labs}')
        else:
            labs = list(self.rev_labels.values())
            print('Inference Mode: Classification')
            print(f'Labels: {labs}')
        print(f'Hardware Acceleration: {self.__device__}')
        print(f'Resources Address: {self.resources_address}')
        if self.embedding_mode == True:
            print(f'Batch Size: {self.batch_size}')
            print(f'Similarity Threshold: {self.embedding_thres}')
            print(f'Clustering Data Address: {self.clustering_data}')
        print('=' * 30)
        return '\n'
    
    def __modelChooser__(self) -> None:
        """
        This method is used to choose best model from w.r.t training log files or model availability

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        dirfiles = os.listdir(f'{self.resources_address}/models')
        if len(dirfiles) == 0:
            raise Exception(f'No Model Weights Found in {self.resources_address}/models')
        elif len(dirfiles) == 1:
            self.classModel = dirfiles[0]
            self.embedModel = dirfiles[0]
        else:
            with open(f'{self.resources_address}/logs/logs', 'rb') as file1:
                self.raw_logs = pickle.load(file1)
            validEpoch = self.raw_logs['epochBased']['validation']['epoch']
            validClass = self.raw_logs['epochBased']['validation']['classificationAccuracy']
            validEmbed = self.raw_logs['epochBased']['validation']['embeddingAccuracy']
            self.classModel = 'model_' + str(validEpoch[np.argmax(validClass)])
            self.embedModel = 'model_' + str(validEpoch[np.argmax(validEmbed)])
    
    def __createCluster__(self) -> None:
        """
        This method is used to create base data cluster for Clustering Mode

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        torch.cuda.empty_cache()
        final_tensor, final_label = list(), list()
        dat = Data(self.clustering_data, transform = inference_transforms)
        self.embedding_classes = dat.rev_classes
        data_loader = torch.utils.data.DataLoader(dat, batch_size=self.batch_size, shuffle=False)
        data_loader_iter = iter(data_loader)
        dat_batches = math.ceil(len(dat) / self.batch_size)
        with tqdm.tqdm(total = dat_batches, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}') as bar:
            for tb, (bdat, blab) in enumerate(data_loader_iter):
                blab = blab.squeeze().type(torch.LongTensor)
                with torch.no_grad():
                    embeds, res = self.mod(bdat.to(self.__device__))
                final_tensor.append(embeds.to('cpu'))
                final_label.append(blab)
                del embeds, res
                torch.cuda.empty_cache()
                bar.set_description('Computing Base Embeddings | Batch')
                bar.update(1)
        self.embedding_tensor = torch.cat(final_tensor).to(self.__device__)
        self.embedding_label = torch.cat(final_label).to(self.__device__)
    
    def __probabilityClassification__(self, bdat) -> list:
        """
        This method is used to classifiy given batch for Classification Mode

        Mehod Input
        ============
        bdat : Batch input data in Torch tensor form
                            Format : [ Batch, Channel, Width, Height ]
        
        Method Output
        ==============
        Inference output list
        """
        outputs = list()
        with torch.no_grad():
            embeds, res = self.mod(bdat.to(self.__device__))
        res = torch.nn.functional.softmax(res, dim = 1).to('cpu')
        maxer = torch.argmax(res, dim=1)
        for i, j in zip(res, maxer):
            outputs.append({
                'Label': self.rev_labels[j.item()],
                'Probability': i[j].item()
            })
        return outputs
    
    def __embeddingClassification__(self, bdat) -> list:
        """
        This method is used to classify given batch for Clustering Mode

        Mehod Input
        ============
        bdat : Batch input data in Torch tensor form
                            Format : [ Batch, Channel, Width, Height ]
        
        Method Output
        ==============
        Inference output list
        """
        with torch.no_grad():
            embeds, res = self.mod(bdat.to(self.__device__))
        sims = torch.nn.functional.cosine_similarity(embeds.unsqueeze(1), self.embedding_tensor.to(self.__device__), dim=2)
        cla1 = torch.unique(self.embedding_label)
        outputs = list()
        for i in sims:
            celem = list()
            cargu = list()
            csimi = list()
            for k in cla1:
                allables = self.embedding_label == k
                tsims = i[allables]
                ttres = tsims >= self.embedding_thres
                celem.append(sum(ttres).item())
                cargu.append(torch.argmax(tsims).item())
                csimi.append(tsims[cargu[-1]].item())
                del k
                torch.cuda.empty_cache()
            tgt_class = np.argmax(celem)
            outputs.append({
                'Label': self.embedding_classes[tgt_class],
                'Probability': csimi[tgt_class]
            })
            del i
            torch.cuda.empty_cache()
        return outputs
    
    def __decodeImage(self, eimgs: list):
        """
        This method is used to decode images to Torch tensor batch

        Method Input
        =============
        eimgs : List of encoded images

        Mathod Output
        ==============
        Torch tensor input batch
        """
        tstack = list()
        for i in eimgs:
            ti = base64.b64decode(i.encode('utf-8'))
            byt1 = BytesIO(ti)
            timg = inference_transforms(Image.open(byt1))
            tstack.append(timg)
        return torch.stack(tstack)
    
    def __call__(self, eimgs: list):
        """
        This method is used to handle inference for both Classification & Clustering Mode

        Method Input
        =============
        eimgs : List of encoded images

        Mathod Output
        ==============
        Inference output list
        """
        torch.cuda.empty_cache()
        dat1 = self.__decodeImage(eimgs)
        if self.embedding_mode == True:
            return self.__embeddingClassification__(dat1)
        else:
            return self.__probabilityClassification__(dat1)

# %%
# Execution
app = Flask(__name__)

@app.post('/')
def runner():
    res1 = ser(request.json)
    return jsonify(res1), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Autism Classification Inference Server.')
    parser.add_argument('-r', '--resources', type = str, help = 'Absolute Address of Trained Resources Directory', default = '/resources')
    parser.add_argument('-thres', '--sim_thres', type = float, help = 'Similarity Threshold in Case of Clustering Mode', default = 0.75)
    parser.add_argument('-btch', '--batch_size', type = int, help = 'Batch Size to Create Base Cluster Embeddings', default = 32)
    parser.add_argument('-data', '--base_data', type = str, help = 'Absolute Address of Base Clustering Data Directory', default = '/data')
    args = vars(parser.parse_args())
    print("""
    ==========================================
    | Autism Classification Inference Server |
    ==========================================
    """)
    ser = Inference(args['resources'], args['sim_thres'], args['batch_size'], args['base_data'])
    print(ser)
    app.run(host='0.0.0.0', port=8080)
