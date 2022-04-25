import torch.nn.functional as F
import torch.nn as nn


# Network definition for the textual aggregation
class EmbeddingTextNN(nn.Module):
    def __init__(self, embedding_size, output_size):
        super(EmbeddingTextNN, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(embedding_size, 1024),
                                 nn.PReLU(),
                                 nn.Linear(1024, 2048),
                                 nn.PReLU(),
                                 nn.Linear(2048, output_size)
                                 )

    # forward method
    def forward(self, x):
        # Project to common latent space for the image and text
        out = self.fc1(x)

        return out


# Network definition for the image embedding
class EmbeddingImageNN(nn.Module):
    def __init__(self, output_size):
        super(EmbeddingImageNN, self).__init__()

        # Define a fully connected layer with input of n_input and output n_output neurons
        self.fc1 = nn.Sequential(nn.Linear(4096, 2048),
                                 nn.PReLU(),
                                 nn.Linear(2048, output_size)
                                 )  # output_size is the size of the final image embedding
        # Define a dropout layer with probability p
        self.dropout = nn.Dropout(p=0.5)

    # forward method
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        return out


# Network definition for the triplet network in the image to text case
class TripletImage2Text(nn.Module):
    def __init__(self, embedding_text_net, embedding_image_net, margin=1.0):
        super(TripletImage2Text, self).__init__()
        self.embedding_text_net = embedding_text_net
        self.embedding_image_net = embedding_image_net
        self.margin = margin

    def forward(self, img, positive_text, negative_text):
        # Get the embeddings for the image and the text
        img_embedding = self.embedding_image_net(img)
        text_embedding = self.embedding_text_net(positive_text)
        negative_text_embedding = self.embedding_text_net(negative_text)

        return img_embedding, text_embedding, negative_text_embedding

    def get_embedding_pair(self, img, text):
        # Get the embeddings for the image and the text
        img_embedding = self.embedding_image_net(img)
        text_embedding = self.embedding_text_net(text)
        return img_embedding, text_embedding


# Network definition for the triplet network in the text to image case
class TripletText2Image(nn.Module):
    def __init__(self, embedding_text_net, embedding_image_net, margin=1.0):
        super(TripletText2Image, self).__init__()
        self.embedding_text_net = embedding_text_net
        self.embedding_image_net = embedding_image_net
        self.margin = margin

    def forward(self, text, img1, img2):
        # Get the embeddings for the image and the text

        text_embedding = self.embedding_text_net(text)
        img1_embedding = self.embedding_image_net(img1)
        img2_embedding = self.embedding_image_net(img2)

        return text_embedding, img1_embedding, img2_embedding

    def get_embedding_pair(self, img, text):
        # Get the embeddings for the image and the text
        img_embedding = self.embedding_image_net(img)
        text_embedding = self.embedding_text_net(text)
        return img_embedding, text_embedding


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
