# from encoder.params_model import model_hidden_size, model_num_layers, model_embedding_size
# from encoder.params_data import mel_n_channels
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch
from t3nsor.layers import TTLinear
from utils.modelutils import count_model_params


class SpeakerEncoder(nn.Module):
    def __init__(self, mel_n_channels, model_hidden_size, model_num_layers,
                 model_embedding_size, device, loss_device, use_tt=False, n_cores=3, tt_rank=8,
                 use_low_rank=False):
        super().__init__()

        if use_tt and use_low_rank:
            raise ValueError("use_tt and use_low_rank cannot both be True.")

        self.loss_device = loss_device
        
        # Network definition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        if not use_tt:
            if not use_low_rank:
                self.linear = nn.Linear(in_features=model_hidden_size,
                                    out_features=model_embedding_size).to(device)
            else:
                # implement low-rank
                raise NotImplementedError()
        else:
            print("Encoding linear layer as a tensor-train...")
            self.linear = TTLinear(in_features=model_hidden_size, out_features=model_embedding_size,
                                   bias=True, auto_shapes=True, d=n_cores, tt_rank=tt_rank).to(device)
        print("Number of parameters in last layer: ", count_model_params(self.linear))

        self.relu = torch.nn.ReLU().to(device)
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds
    
    def similarity_matrix(self, verification_embeds, enrollment_embeds=None):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param verification_embeds: the verification embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :param enrollment_embeds: the enrollment embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size). If None (eg during training), uses verification_embeds for enrollment.
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = verification_embeds.shape[:2]
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(verification_embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        if enrollment_embeds is None:
            # Exclusive centroids (1 per utterance)
            centroids_excl = (torch.sum(verification_embeds, dim=1, keepdim=True) - verification_embeds)
            centroids_excl /= (utterances_per_speaker - 1)
            centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

            # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
            # product of these vectors (which is just an element-wise multiplication reduced by a sum).
            mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
            for j in range(speakers_per_batch):
                mask = np.where(mask_matrix[j])[0]
                sim_matrix[mask, :, j] = (verification_embeds[mask] * centroids_incl[j]).sum(dim=2)
                sim_matrix[j, :, j] = (verification_embeds[j] * centroids_excl[j]).sum(dim=1)
        else:
            centroids = torch.mean(enrollment_embeds, dim=1, keepdim=True)
            centroids = centroids.clone() / torch.norm(centroids, dim=2, keepdim=True)
            for j in range(speakers_per_batch):
                sim_matrix[:, :, j] = (verification_embeds * centroids[j, :, :]).sum(dim=2)
       
        # print("similarity weight: {}\t similarity bias: {}\tweight_grad: {}\tbias_grad: {}"
                # .format(self.similarity_weight.item(), self.similarity_bias.item(),
                    # self.similarity_weight.grad, self.similarity_bias.grad))

        # print("No of Nans in similarity matrix", torch.sum(sim_matrix != sim_matrix))
        # print("No of Nans in verification embeds", torch.sum(verification_embeds != verification_embeds))
        # if enrollment_embeds is not None:
        #    print("No of Nans in enrollment embeds", torch.sum(enrollment_embeds != enrollment_embeds))
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, verification_embeds, enrollment_embeds=None):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param verification_embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = verification_embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(verification_embeds, enrollment_embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer


class LRLinear(nn.Module):
    """
    Low-rank factorised linear module.
    """
    def __init__(self, in_features, out_features, rank, device, bias=True):
        super(LRLinear, self).__init__()

        self.linear1 = nn.Linear(in_features=in_features,
                                    out_features=rank, bias=bias).to(device)
        self.linear2 = nn.Linear(in_features=rank,
                                 out_features=out_features, bias=bias).to(device)

    def forward(self, x):
        return self.linear2(self.linear1(x))
