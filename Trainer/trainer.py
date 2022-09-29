#!/usr/bin/env python3

"""
MODEL TRAINER
=============
Following program is used to train model on updated or new data
"""

# %%
# Importing libraries
from model import *
import string
import math
import copy
import argparse
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
import seaborn as sns

# %%
# Main Data Loading Class
class Data(torch.utils.data.Dataset):
    def __init__(self, addr: string, classes: dict, transform, shuffle: bool=True) -> None:
        """
        This method is used to initialize data loading class
        
        Method Input
        =============
        addr : List containing absolute address of files to include as data
        classes : Dictionary containing respective Ohe Hot Encoded labels for images list
        transform : Subject transforms to apply on the data
        shuffle : Data Shuffling
                            ( Default : True )
        
        Method Output
        ==============
        None
        """
        super(Data, self).__init__()
        self.data_addr = addr
        self.classes = classes
        self.transform = transform
        self.shuffle = shuffle
        self.ohe_classes = {j:i for i, j in self.classes.items()}
        self.data_samples, self.data_labels = list(), list()
        for i, j in self.classes.items():
            for k in os.listdir(f'{self.data_addr}/{i}'):
                self.data_samples.append(f'{self.data_addr}/{i}/{k}')
                self.data_labels.append(j)
        if self.shuffle == True:
            shuf = np.arange(len(self.data_labels))
            np.random.shuffle(shuf)
            self.data_samples = np.array(self.data_samples)[shuf].tolist()
            self.data_labels = np.array(self.data_labels)[shuf].tolist()
    
    def __len__(self) -> int:
        """
        This method is used to find the number of files in respective dataset
        
        Method Input
        =============
        None
        
        Method Output
        ==============
        None
        """
        return len(self.data_labels)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        This method is used to load and process the image based on file number
        
        Method Input
        =============
        idx : File number ( 0 -> self.__len__())
        
        Method Output
        ==============
        Processed Image Data, Respective Image One Hot Encoded Label
        """
        img = Image.open(self.data_samples[idx])
        timg = self.transform(img)
        return timg, torch.Tensor([self.data_labels[idx]]).type(torch.int32)

# %%
# Main Trainer Class
class Trainer:
    def __init__(self, addr: string,
                        epochs: int = 5,
                        learn_rate: float = 0.00001,
                        batch_size: int = 32,
                        train_shuffle: bool = True,
                        sim_thres: float = 0.75,
                        train_metric: bool = False,
                        seed: int = 42,
                        outdir: string = '/output') -> None:
        """
        This method is used to initialize trainer class

        Method Input
        =============
        addr : Absolute address of data directory for training
                            Default Directory Hierarchy:

                                data
                                  |_ training
                                  |       |_ class-1
                                  |       |_ class-2
                                  |       |_ ...
                                  |
                                  |_ validation
                                  |       |_ class-1
                                  |       |_ class-2
                                  |       |_ ...
                                  |
                                  |_ testing
                                          |_ class-1
                                          |_ class-2
                                          |_ ...
        
        epochs : Number of epochs used for training ( Default : 5 )
        learn_rate : Learning rate for training ( Default : 0.001 )
        batch_size : Batch size for training ( Default : 32 )
        train_shuffle : Choice to shuffle training data during training process  ( Default : True )
        sim_thres : Similarity threshold value used for computing embeddings accuracy ( Default : 0.75 )
        seed : Seed value for training data shuffle
        outdir : Output directory to save training outputs

        Method Output
        ==============
        None
        """
        torch.cuda.empty_cache()
        self.dataset_address = addr
        self.epochs = epochs
        self.learning_rate = learn_rate
        self.batch_size = batch_size
        self.training_shuffle = train_shuffle
        self.similarity_thres = sim_thres
        self.training_metrics = train_metric
        self.seed = seed
        self.output_directory = outdir
        self.__device__ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.training_data_address = f'{self.dataset_address}/training'
        self.validation_data_address = f'{self.dataset_address}/validation'
        self.testing_data_address = f'{self.dataset_address}/testing'
        self.classes = {j:i for i, j in enumerate(os.listdir(self.training_data_address))}
        self.sample_breakdown = {'Training':dict(), 'Validation': dict(), 'Testing':dict()}
        with open(f'{self.output_directory}/labels', 'wb') as file1:
            pickle.dump(self.classes, file1)
        self.rev_classes = {j:i for i, j in self.classes.items()}
        for i in list(self.classes.keys()):
            train_files = os.listdir(f'{self.training_data_address}/{i}')
            valid_files = os.listdir(f'{self.validation_data_address}/{i}')
            test_files = os.listdir(f'{self.testing_data_address}/{i}')
            if len(train_files) == 0:
                raise Exception(f'Training Data with {i} Class has no Samples')
            else:
                self.sample_breakdown['Training'][i] = len(train_files)
            if  len(valid_files) == 0:
                raise Exception(f'Validation Data with {i} Class has no Samples')
            else:
                self.sample_breakdown['Validation'][i] = len(valid_files)
            if len(test_files) == 0:
                raise Exception(f'Testing Data with {i} Class has no Samples')
            else:
                self.sample_breakdown['Testing'][i] = len(test_files)
        os.mkdir(f'{self.output_directory}/logs')
        os.mkdir(f'{self.output_directory}/models')
        self.__history__ = {
            'live': {
                'cumulative': {'epoch': list(), 'batch': list(), 'loss': list(), 'classificationAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()},
                'epochBased': {'epoch': list(), 'loss': list(), 'classificationAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()}
            },
            'cumulative': {
                'training': {'epoch': list(), 'batch': list(), 'loss': list(), 'classificationAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()},
                'validation': {'epoch': list(), 'batch': list(), 'loss': list(), 'classificationAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()},
                'testing': {'epoch': list(), 'batch': list(), 'loss': list(), 'classificationAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()}
            },
            'epochBased': {
                'training': {'epoch': list(), 'loss': list(), 'classificationAccuracy': list(), 'embeddingAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()},
                'validation': {'epoch': list(), 'loss': list(), 'classificationAccuracy': list(), 'embeddingAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()},
                'testing': {'epoch': list(), 'loss': list(), 'classificationAccuracy': list(), 'embeddingAccuracy': list(), 'precision': list(), 'recall': list(), 'f1_score': list(), 'jaccard': list()}
            }
        }
        self.__data_process__()
    
    def __str__(self) -> None:
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
        print(f'Epochs: {self.epochs}')
        print(f'Learning Rate: {self.learning_rate}')
        print(f'Batch Size: {self.batch_size}')
        print(f'Similarity Threshold: {self.similarity_thres}')
        print(f'Training Shuffle: {self.training_shuffle}')
        print(f'Training Batches: {self.train_batches}')
        print(f'Validation Batches: {self.valid_batches}')
        print(f'Testing Batches: {self.test_batches}')
        print(f'Extensive Training Metrics Computation: {self.training_metrics}')
        print(f'Shuffle Seed Value: {self.seed}')
        print(f'Hardware Acceleration: {self.__device__}')
        print(f'Dataset Directory: {self.dataset_address}')
        print(f'Output Directory: {self.output_directory}')
        print(f'Sample Breakdown: {self.sample_breakdown}')
        print('=' * 30)
        return '\n'
    
    def __data_process__(self) -> None:
        """
        This method is used to process data for training process

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        self.train_data = Data(self.training_data_address, self.classes, training_transforms, self.training_shuffle)
        self.train_metric_data = Data(self.training_data_address, self.classes, inference_transforms, False)
        self.valid_data = Data(self.validation_data_address, self.classes, inference_transforms, False)
        self.test_data = Data(self.testing_data_address, self.classes, inference_transforms, False)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.training_shuffle)
        self.train_mertic_loader = torch.utils.data.DataLoader(self.train_metric_data, batch_size=self.batch_size, shuffle=False)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        self.train_batches = math.ceil(len(self.train_data) / self.batch_size)
        self.metrc_batches = math.ceil(len(self.train_metric_data) / self.batch_size)
        self.valid_batches = math.ceil(len(self.valid_data) / self.batch_size)
        self.test_batches = math.ceil(len(self.test_data) / self.batch_size)
    
    def __write_logs__(self) -> None:
        """
        This method is used to write training logs to file

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        hist = f',,,,,,Autism Classification Training Logs,,,,,\n\n'
        hist += f'Metadata\n'
        hist += f',Epochs,{self.epochs}\n'
        hist += f',Learning Rate,{self.learning_rate}\n'
        hist += f',Batch Size,{self.batch_size}\n'
        hist += f',Similarity Threshold,{self.similarity_thres}\n'
        hist += f',Seed Value,{self.seed}\n'
        hist += f',Training Data Shuffle,{self.training_shuffle}\n\n'
        hist += f',Training Samples,{len(self.train_data)}\n'
        hist += f',Validation Samples,{len(self.valid_data)}\n'
        hist += f',Testing Samples,{len(self.test_data)}\n\n'
        hist += f'Classes\n,Class Name,One Hot Encoding,Training Samples,Validation Samples,Testing Samples\n'
        for i, j in self.classes.items():
            tr = self.sample_breakdown['Training'][i]
            va = self.sample_breakdown['Validation'][i]
            te = self.sample_breakdown['Testing'][i]
            hist += f',{i},{j},{tr},{va},{te}\n'
        hist += '\nLive Training Logs\n'
        hist += ',Cumulative\n'
        hist += ',Epoch,Batch,Loss,Classification Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['live']['cumulative']['epoch'], self.__history__['live']['cumulative']['batch'],
                                            self.__history__['live']['cumulative']['loss'], self.__history__['live']['cumulative']['classificationAccuracy'],
                                            self.__history__['live']['cumulative']['precision'], self.__history__['live']['cumulative']['recall'],
                                            self.__history__['live']['cumulative']['f1_score'], self.__history__['live']['cumulative']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        hist += '\n,Epoch Based\n'
        hist += ',Epoch,Loss,Classification Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, k, l, m, n, o, p in zip(self.__history__['live']['epochBased']['epoch'],
                                            self.__history__['live']['epochBased']['loss'], self.__history__['live']['epochBased']['classificationAccuracy'],
                                            self.__history__['live']['epochBased']['precision'], self.__history__['live']['epochBased']['recall'],
                                            self.__history__['live']['epochBased']['f1_score'], self.__history__['live']['epochBased']['jaccard']):
            hist += f',{i},{k},{l},{m},{n},{o},{p}\n'
        hist += '\nCumulative\n'
        hist += ',Training\n'
        hist += ',Epoch,Batch,Loss,Classification Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['cumulative']['training']['epoch'], self.__history__['cumulative']['training']['batch'],
                                            self.__history__['cumulative']['training']['loss'], self.__history__['cumulative']['training']['classificationAccuracy'],
                                            self.__history__['cumulative']['training']['precision'], self.__history__['cumulative']['training']['recall'],
                                            self.__history__['cumulative']['training']['f1_score'], self.__history__['cumulative']['training']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        hist += '\n,Validation\n'
        hist += ',Epoch,Batch,Loss,Classification Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['cumulative']['validation']['epoch'], self.__history__['cumulative']['validation']['batch'],
                                            self.__history__['cumulative']['validation']['loss'], self.__history__['cumulative']['validation']['classificationAccuracy'],
                                            self.__history__['cumulative']['validation']['precision'], self.__history__['cumulative']['validation']['recall'],
                                            self.__history__['cumulative']['validation']['f1_score'], self.__history__['cumulative']['validation']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        hist += '\n,Testing\n'
        hist += ',Epoch,Batch,Loss,Classification Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['cumulative']['testing']['epoch'], self.__history__['cumulative']['testing']['batch'],
                                            self.__history__['cumulative']['testing']['loss'], self.__history__['cumulative']['testing']['classificationAccuracy'],
                                            self.__history__['cumulative']['testing']['precision'], self.__history__['cumulative']['testing']['recall'],
                                            self.__history__['cumulative']['testing']['f1_score'], self.__history__['cumulative']['testing']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        hist += 'Epoch Based\n'
        hist += '\n,Training\n'
        hist += ',Epoch,Loss,Classification Accuracy,Embedding Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['loss'],
                                            self.__history__['epochBased']['training']['classificationAccuracy'], self.__history__['epochBased']['training']['embeddingAccuracy'],
                                            self.__history__['epochBased']['training']['precision'], self.__history__['epochBased']['training']['recall'],
                                            self.__history__['epochBased']['training']['f1_score'], self.__history__['epochBased']['training']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        hist += '\n,Validation\n'
        hist += ',Epoch,Loss,Classification Accuracy,Embedding Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['loss'],
                                            self.__history__['epochBased']['validation']['classificationAccuracy'], self.__history__['epochBased']['validation']['embeddingAccuracy'],
                                            self.__history__['epochBased']['validation']['precision'], self.__history__['epochBased']['validation']['recall'],
                                            self.__history__['epochBased']['validation']['f1_score'], self.__history__['epochBased']['validation']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        hist += '\n,Testing\n'
        hist += ',Epoch,Loss,Classification Accuracy,Embedding Accuracy,Precision,Recall,F1 Score,Jaccard Score\n'
        for i, j, k, l, m, n, o, p in zip(self.__history__['epochBased']['testing']['epoch'], self.__history__['epochBased']['testing']['loss'],
                                            self.__history__['epochBased']['testing']['classificationAccuracy'], self.__history__['epochBased']['testing']['embeddingAccuracy'],
                                            self.__history__['epochBased']['testing']['precision'], self.__history__['epochBased']['testing']['recall'],
                                            self.__history__['epochBased']['testing']['f1_score'], self.__history__['epochBased']['testing']['jaccard']):
            hist += f',{i},{j},{k},{l},{m},{n},{o},{p}\n'
        with open(f'{self.output_directory}/logs/logs.csv', 'w') as file1:
            file1.write(hist)
        with open(f'{self.output_directory}/logs/logs', 'wb') as file1:
            pickle.dump(self.__history__, file1)

    def __plot_histogram__(self, saveAddr: string,
                                    true_probs: list,
                                    false_probs: list,
                                    plt_title: string,
                                    plt_xlabel: string = 'Similarity Score',
                                    plt_ylabel: string = 'Density') -> None:
        """
        This method is used to plot & save respective logs histograms
        
        Method Input
        =============
        saveAddr : Absolute address to save histogram plot
        true_probs : Probabilities of true class
        false_probs : Probabilities of false class
        plt_title : Plot title to to display in histogram plot
        plt_xlabel : x-axis label in histogram plot
        plt_ylabel : y-axis label in histgram plot

        Method Output
        ==============
        None
        """
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        sns.distplot(true_probs, hist = False, kde = True, rug = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'True')
        sns.distplot(false_probs, hist = False, kde = True, rug = True, kde_kws = {'shade': True, 'linewidth': 3}, label = 'False')
        plt.legend(prop={'size': 16}, title = 'Density')
        plt.title(plt_title)
        plt.xlabel(plt_xlabel)
        plt.ylabel(plt_ylabel)
        plt.xlim([0.0, 1.0])
        plt.savefig(saveAddr)
        plt.close('all')
    
    def __plot_confusion__(self, saveAddr: string, actual: list, predicted: list, plt_title: string) -> None:
        """
        This method is used to plot & save respective logs confusion matrix

        Method Input
        =============
        saveAddr : Absolute address to save confusino matrix
        actual : Actual class labels
        predicted : Predicted class labels
        plt_title : Plot title to to display in confusion matrix plot

        Method Output
        ==============
        None
        """
        confus = confusion_matrix(actual, predicted)
        dfr = pd.DataFrame(confus, list(self.classes.keys()), list(self.classes.keys()))
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plotter = sns.heatmap(dfr, cmap='viridis', annot=True, fmt='d', cbar=False).get_figure()
        plt.title(plt_title)
        plt.savefig(saveAddr)
        plt.close('all')
    
    def __plot__history__(self) -> None:
        """
        This method is used to plot respective logs plot

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['live']['epochBased']['epoch'], self.__history__['live']['epochBased']['loss'], label='Live Training Loss')
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['loss'], label='Training Loss')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['loss'], label='Validation Loss')
        if len(self.__history__['epochBased']['testing']['loss']) !=0:
            plt.plot([], [], '', label='Testing Loss: ' + str(self.__history__['epochBased']['testing']['loss'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'{self.output_directory}/logs/epoch_loss.png')
        plt.close('all')
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['live']['epochBased']['epoch'], self.__history__['live']['epochBased']['classificationAccuracy'], label='Live Training Accuracy')
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['classificationAccuracy'], label='Training Accuracy')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['classificationAccuracy'], label='Validation Accuracy')
        if len(self.__history__['epochBased']['testing']['classificationAccuracy']) !=0:
            plt.plot([], [], '', label='Testing Accuracy: ' + str(self.__history__['epochBased']['testing']['classificationAccuracy'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch Clasification Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(f'{self.output_directory}/logs/epoch_classification_accuracy.png')
        plt.close('all')
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['embeddingAccuracy'], label='Training Accuracy')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['embeddingAccuracy'], label='Validation Accuracy')
        if len(self.__history__['epochBased']['testing']['embeddingAccuracy']) !=0:
            plt.plot([], [], '', label='Testing Accuracy: ' + str(self.__history__['epochBased']['testing']['embeddingAccuracy'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch Embedding Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(f'{self.output_directory}/logs/epoch_embedding_accuracy.png')
        plt.close('all')
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['live']['epochBased']['epoch'], self.__history__['live']['epochBased']['precision'], label='Live Training Precision')
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['precision'], label='Training Precision')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['precision'], label='Validation Precision')
        if len(self.__history__['epochBased']['testing']['precision']) !=0:
            plt.plot([], [], '', label='Testing Precision: ' + str(self.__history__['epochBased']['testing']['precision'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.savefig(f'{self.output_directory}/logs/epoch_precision.png')
        plt.close('all')
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['live']['epochBased']['epoch'], self.__history__['live']['epochBased']['recall'], label='Live Training Recall')
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['recall'], label='Training Recall')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['recall'], label='Validation Recall')
        if len(self.__history__['epochBased']['testing']['recall']) !=0:
            plt.plot([], [], '', label='Testing Recall: ' + str(self.__history__['epochBased']['testing']['recall'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.savefig(f'{self.output_directory}/logs/epoch_recall.png')
        plt.close('all')
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['live']['epochBased']['epoch'], self.__history__['live']['epochBased']['f1_score'], label='Live Training F1-Score')
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['f1_score'], label='Training F1-Score')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['f1_score'], label='Validation F1-Score')
        if len(self.__history__['epochBased']['testing']['f1_score']) !=0:
            plt.plot([], [], '', label='Testing F1-Score: ' + str(self.__history__['epochBased']['testing']['f1_score'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch F1-Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score')
        plt.savefig(f'{self.output_directory}/logs/epoch_f1score.png')
        plt.close('all')
        plt.clf()
        plt.figure(figsize = ([13.1, 7.1]))
        plt.plot(self.__history__['live']['epochBased']['epoch'], self.__history__['live']['epochBased']['jaccard'], label='Live Training Jaccard Score')
        plt.plot(self.__history__['epochBased']['training']['epoch'], self.__history__['epochBased']['training']['jaccard'], label='Training Jaccard Score')
        plt.plot(self.__history__['epochBased']['validation']['epoch'], self.__history__['epochBased']['validation']['jaccard'], label='Validation Jaccard Score')
        if len(self.__history__['epochBased']['testing']['jaccard']) !=0:
            plt.plot([], [], '', label='Testing Jaccard Score: ' + str(self.__history__['epochBased']['testing']['jaccard'][-1]))
        plt.legend(title = 'Metadata')
        plt.title('Epoch Jaccard Score')
        plt.xlabel('Epochs')
        plt.ylabel('Jaccard Score')
        plt.savefig(f'{self.output_directory}/logs/epoch_jaccard.png')
        plt.close('all')

    def __tester__(self,current_epoch: int, key: string, bar) -> None:
        """
        This method is used to compute respective metrics against some dataset

        Method Input
        =============
        current_epoch : Current epoch in training process
        key : Data key to compute metrics for
                            Choices : [ training, validation, testing ]
        bar : Display bar to show live metrics computation & logs

        Method Output
        ==============
        None
        """
        torch.cuda.empty_cache()
        self.mod.eval()
        savKey = key.capitalize()
        tloss, eloss = 0, 0
        ecorrect = 0
        raw_embeds = {i: list() for i in list(self.classes.keys())}
        raw_classe = {i: list() for i in list(self.classes.keys())}
        raw_probab = {i: list() for i in list(self.classes.keys())}
        raw_eclass = {i: list() for i in list(self.classes.keys())}
        raw_eproba = {i: list() for i in list(self.classes.keys())}
        act_classe = {i: list() for i in list(self.classes.keys())}
        tcprobs, fcprobs = list(), list()
        teprobs, feprobs = list(), list()
        tpred, tact = list(), list()
        if key == 'training':
            iterat = iter(self.train_mertic_loader)
            bar.reset(total=self.train_batches)
        if key == 'validation':
            iterat = iter(self.valid_data_loader)
            bar.reset(total=self.valid_batches)
        if key == 'testing':
            iterat = iter(self.test_data_loader)
            bar.reset(total=self.test_batches)
        for tb, (dat, labs) in enumerate(iterat):
            labs = labs.squeeze().type(torch.LongTensor)
            with torch.no_grad():
                embeds, res = self.mod(dat.to(self.__device__))
                loss1 = self.loss(res, labs.to(self.__device__))
            embeds = embeds.to('cpu')
            res = torch.nn.functional.softmax(res, dim=1).to('cpu')
            loss1 = loss1.to('cpu')
            lab_res = torch.argmax(res, dim=1)
            prob_res = torch.Tensor([i[j].item() for i, j in zip(res, lab_res)])
            self.__history__['cumulative'][key]['epoch'].append(current_epoch)
            self.__history__['cumulative'][key]['batch'].append(tb)
            self.__history__['cumulative'][key]['loss'].append(loss1.item())
            self.__history__['cumulative'][key]['classificationAccuracy'].append(sum(lab_res == labs) / len(labs))
            self.__history__['cumulative'][key]['precision'].append(precision_score(labs.numpy(), lab_res.numpy(), average='micro'))
            self.__history__['cumulative'][key]['recall'].append(recall_score(labs.numpy(), lab_res.numpy(), average='micro'))
            self.__history__['cumulative'][key]['f1_score'].append(f1_score(labs.numpy(), lab_res.numpy(), average='micro'))
            self.__history__['cumulative'][key]['jaccard'].append(jaccard_score(labs.numpy(), lab_res.numpy(), average='micro'))
            tloss += self.__history__['cumulative'][key]['loss'][-1]
            for k in list(self.classes.values()):
                activ = torch.where(labs==k, True, False)
                raw_embeds[self.rev_classes[k]].extend(embeds[activ].tolist())
                raw_classe[self.rev_classes[k]].extend(lab_res[activ].tolist())
                raw_probab[self.rev_classes[k]].extend(prob_res[activ].tolist())
                act_classe[self.rev_classes[k]].extend(labs[activ].tolist())
            del labs, res, embeds, lab_res, prob_res, loss1
            disp_loss = tloss / (tb + 1)
            disp_accu = self.__history__['cumulative'][key]['classificationAccuracy'][-1]
            bar.set_description('{} Metrics | Loss: {:.5f} \\ Accuracy: {:.5f} | Batch'.format(savKey, disp_loss, disp_accu))
            bar.update(1)
        for l in list(self.classes.keys()):
            tpred.extend(raw_classe[l])
            tact.extend(act_classe[l])
        torch.cuda.empty_cache()
        eloss = tloss / (tb + 1)
        res_accu = sum(np.array(tpred) == np.array(tact)) / len(tact)
        self.__history__['epochBased'][key]['epoch'].append(current_epoch)
        self.__history__['epochBased'][key]['loss'].append(eloss)
        self.__history__['epochBased'][key]['classificationAccuracy'].append(res_accu)
        self.__history__['epochBased'][key]['precision'].append(precision_score(tact, tpred, average='micro'))
        self.__history__['epochBased'][key]['recall'].append(recall_score(tact, tpred, average='micro'))
        self.__history__['epochBased'][key]['f1_score'].append(f1_score(tact, tpred, average='micro'))
        self.__history__['epochBased'][key]['jaccard'].append(jaccard_score(tact, tpred, average='micro'))
        bar.reset(total=len(tact))
        for m in list(raw_embeds.keys()):
            for n in range(len(raw_embeds[m])):
                temp_embeds = copy.deepcopy(raw_embeds)
                temp_alabs = copy.deepcopy(act_classe)
                temp_summer, temp_maxer, temp_proba = dict(), dict(), dict()
                tgt_tensor = temp_embeds[m][n]
                del temp_embeds[m][n]
                del temp_alabs[m][n]
                for o in list(temp_embeds.keys()):
                    temp_embeds[o] = torch.nn.functional.cosine_similarity(torch.Tensor(tgt_tensor).to(self.__device__), torch.Tensor(temp_embeds[o]).to(self.__device__)).to('cpu')
                    temp_summer[o] = sum(temp_embeds[o]>=self.similarity_thres).item()
                    temp_maxer[o] = torch.argmax(temp_embeds[o]).item()
                    temp_proba[o] = temp_embeds[o][temp_maxer[o]]
                maxer = np.argmax(list(temp_summer.values()))
                if m==list(temp_summer.keys())[maxer]:
                    ecorrect += 1
                raw_eclass[m].append(np.argmax(list(temp_proba.values())))
                raw_eproba[m].append(list(temp_proba.values())[raw_eclass[m][-1]].item())
                bar.set_description(f'{savKey} Metrics | Contrastive Classification Sample')
                bar.update(1)
        self.__history__['epochBased'][key]['embeddingAccuracy'].append(ecorrect / len(tact))
        for p in list(act_classe.keys()):
            for q, r, s, t, u in zip(act_classe[p], raw_classe[p], raw_probab[p], raw_eclass[p], raw_eproba[p]):
                if q == r:
                    tcprobs.append(s)
                else:
                    fcprobs.append(s)
                if q == t:
                    teprobs.append(u)
                else:
                    feprobs.append(u)
        self.__plot_confusion__(f'{self.output_directory}/logs/{key}_{current_epoch}_confusion_matrix.png', tact, tpred, f'{savKey}:{current_epoch} Confusion Matrix')
        self.__plot_histogram__(f'{self.output_directory}/logs/{key}_{current_epoch}_classification_histogram.png', tcprobs, fcprobs, f'{savKey}:{current_epoch} Classification Histogram')
        self.__plot_histogram__(f'{self.output_directory}/logs/{key}_{current_epoch}_embedding_histogram.png', teprobs, feprobs, f'{savKey}:{current_epoch} Embedding Histogram')
    
    def __train_epoch__(self, current_epoch: int, tr_bar) -> None:
        """
        This method is used to perform one training epoch

        Method Input
        =============
        current_epoch : Current epoch in training process
        bar : Display bar to show live training metrics & logs

        Method Output
        ==============
        None
        """
        tr_bar.reset(total=self.train_batches)
        self.mod.train()
        self.train_data_loader_iter = iter(self.train_data_loader)
        act_labs, pred_labs, tloss = list(), list(), 0
        for tb, (dat, labs) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            labs = labs.squeeze().type(torch.LongTensor)
            with torch.cuda.amp.autocast():
                embeds, res = self.mod(dat.to(self.__device__))
                loss1 = self.loss(res, labs.to(self.__device__))
            current_loss = loss1.item()
            self.grad_scaler.scale(loss1).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            tloss += current_loss
            lab_res = torch.argmax(res, dim=1).detach().to('cpu').numpy()
            labs = labs.detach().to('cpu').numpy()
            act_labs.extend(labs.tolist())
            pred_labs.extend(lab_res.tolist())
            self.__history__['live']['cumulative']['epoch'].append(current_epoch)
            self.__history__['live']['cumulative']['batch'].append(tb)
            self.__history__['live']['cumulative']['loss'].append(current_loss)
            self.__history__['live']['cumulative']['classificationAccuracy'].append(sum(lab_res == labs) / len(labs))
            self.__history__['live']['cumulative']['precision'].append(precision_score(labs, lab_res, average='micro'))
            self.__history__['live']['cumulative']['recall'].append(recall_score(labs, lab_res, average='micro'))
            self.__history__['live']['cumulative']['f1_score'].append(f1_score(labs, lab_res, average='micro'))
            self.__history__['live']['cumulative']['jaccard'].append(jaccard_score(labs, lab_res, average='micro'))
            disp_loss = tloss / (tb + 1)
            disp_accu = sum(np.array(act_labs) == np.array(pred_labs)) / len(act_labs)
            tr_bar.set_description('Training | Loss: {:.5f} \\ Accuracy: {:.5f} | Batch'.format(disp_loss, disp_accu))
            tr_bar.update(1)
        self.__history__['live']['epochBased']['epoch'].append(current_epoch)
        self.__history__['live']['epochBased']['loss'].append(tloss / (tb+1))
        self.__history__['live']['epochBased']['classificationAccuracy'].append(sum(np.array(act_labs)==np.array(pred_labs))/len(act_labs))
        self.__history__['live']['epochBased']['precision'].append(precision_score(act_labs, pred_labs, average='micro'))
        self.__history__['live']['epochBased']['recall'].append(recall_score(act_labs, pred_labs, average='micro'))
        self.__history__['live']['epochBased']['f1_score'].append(f1_score(act_labs, pred_labs, average='micro'))
        self.__history__['live']['epochBased']['jaccard'].append(jaccard_score(act_labs, pred_labs, average='micro'))
        torch.save(self.mod.state_dict(), f'{self.output_directory}/models/model_{current_epoch}')   

    def __call__(self) -> None:
        """
        This method is used to execute & handle training process

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        self.mod = Model(len(self.classes))
        self.mod.to(self.__device__)
        self.optimizer = torch.optim.Adam(self.mod.parameters(), lr =self.learning_rate, weight_decay=1e-4)
        with tqdm.tqdm(total = self.epochs, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 1) as bar:
            with tqdm.tqdm(total = self.train_batches, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 0, leave = True) as tr_bar:
                bar.set_description(f'Epoch')
                for e in range(self.epochs):
                    self.__train_epoch__(e, tr_bar)
                    if self.training_metrics == True:
                        self.__tester__(e, 'training', tr_bar)
                    self.__tester__(e, 'validation', tr_bar)
                    self.__write_logs__()
                    self.__plot__history__()
                    bar.set_description('Training > L:{:.5f} \\ A:{:.5f} | Validation > L:{:.5f} \\ A:{:.5f} | Epoch'.format(self.__history__['live']['epochBased']['loss'][-1],
                                                                                                                            self.__history__['live']['epochBased']['classificationAccuracy'][-1],
                                                                                                                            self.__history__['epochBased']['validation']['loss'][-1],
                                                                                                                            self.__history__['epochBased']['validation']['classificationAccuracy'][-1]))
                    bar.update(1)
                self.__tester__(e, 'testing', tr_bar)
                self.__write_logs__()
                self.__plot__history__()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Autism Classification Model Trainer.')
    parser.add_argument('-e', '--epochs', type = int, help = 'Number of Epochs to Which Model Should be Trained Upto', required = True)
    parser.add_argument('-l', '--lr', type = float, help = 'Learning Rate for Model Training', default = 0.00001)
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch Size of Input Data', default = 32)
    parser.add_argument('-ts', '--train_shuffle', type = bool, help = 'Boolean Valiable to Shuffle Training Data ( True / False )', default = True)
    parser.add_argument('-sim', '--sim_thres', type = float, help = 'Similarity Threshold for Metrics Computation', default = 0.75)
    parser.add_argument('-tm', '--train_metric', action = 'store_true', help = 'Flag to Compute Extensive Training Metrics')
    parser.add_argument('-sd', '--seed', type = int, help = 'Seed Value to Randomize Dataset', default = 42)
    parser.add_argument('-d', '--data', type = str, help = 'Absolute Aaddress of the Parent Directory of Data Distribution Sub-Directories', default = '/data')
    parser.add_argument('-tra', '--traddr', type = str, help = 'Absolute Address to Save Training Output', default = '/resources')
    args = vars(parser.parse_args())
    tra = Trainer(args['data'], args['epochs'], args['lr'], args['batch_size'], args['train_shuffle'], args['sim_thres'], args['train_metric'], args['seed'], args['traddr'])
    print(tra)
    tra()
