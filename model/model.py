import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import torchvision.models as models

torch.manual_seed(1)

class EncoderCNN(nn.Module):
    def __init__(self, resnet_out_size, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        resnet_module = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*resnet_module)
        self.linear = nn.Linear(resnet_out_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class EncoderCNN_saved(nn.Module):
    def __init__(self, resnet_out_size, embed_size):
        super(EncoderCNN_saved, self).__init__()
        ## vacancy for the resnet weights used when inferring from raw images
        resnet = models.resnet152(pretrained=True)
        resnet_module = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*resnet_module)
        ##
        self.linear = nn.Linear(resnet_out_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, features):
        """Extract the image feature vectors."""
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, tag_size, num_layers, word2vec, is_ko):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size+tag_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        '''
        if is_ko is True:
            self.embed = nn.Embedding(vocab_size, embed_size)
        else:
            w_emb = torch.FloatTensor(word2vec)
            self.embed = nn.Embedding.from_pretrained(w_emb, freeze=False)
        '''
    def forward(self, features, tags, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        hiddens = self.custom_LSTM(embeddings, tags)
        packed = pack_padded_sequence(hiddens, lengths, batch_first=True)
        outputs = self.linear(packed[0])
        return outputs


    def custom_LSTM(self, inputs, tags, hidden=None):
        steps = inputs.shape[1]
        batch_size = inputs.shape[0]
        outputs = Variable(torch.zeros(batch_size, steps, self.hidden_size)).cuda()

        for i in range(steps):
            input = inputs[:, i, :]
            input_tag = torch.cat((input, tags), 1)
            output, hidden = self.step(input_tag, hidden)
            outputs[:, i, :] = output
        return outputs


    def step(self, input, hidden=None):
        input = input.unsqueeze(1)
        output, hidden = self.lstm(input, hidden)
        output = output.squeeze(1)
        return output, hidden


    def sample(self, features, tags, hidden=None):
        """Samples captions for given image features."""
        sampled_ids = []
        inputs = features

        for i in range(20):  # maximum sampling length
            input_tag = torch.cat((inputs, tags), 1)
            outputs, hidden = self.step(input_tag, hidden)  # (batch_size, 1, hidden_size),
            outputs = self.linear(outputs)  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.stack(sampled_ids)
        sampled_ids = torch.t(sampled_ids)
        return sampled_ids


    def sample_beam(self, features, tags, beam_size=5, hidden=None, max_step=20):
        """Samples captions for given image features with beam search."""
        '''
        def calc_prob(y):
            maxy, idx = torch.max(y[0], 0)
            e = (y[0] - maxy).exp()
            p = e / e.sum()
            tens = (1e-20 + p).log()
            out = torch.tensor([tens.data.cpu().numpy()]).cuda()

            return out
        '''
        inputs = features
        input_tag = torch.cat((inputs, tags), 1)
        outputs, hidden = self.step(input_tag, hidden)
        outputs = self.linear(outputs)
        logprobs = log_softmax(outputs, dim=1)
        #logprobs = calc_prob(outputs)
        logprobs, predicteds = logprobs.topk(beam_size, 1)

        beams = []
        nsteps = 1
        for i in range(beam_size):
            beams.append((logprobs[:, i], [predicteds[:, i]], hidden))

        while True:
            beam_candidates = []
            for b in beams:

                if b[1][-1] == 0:
                    # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue

                input_tag = torch.cat((self.embed(b[1][-1]), tags), 1)
                outputs, hidden = self.step(input_tag, b[2])
                outputs = self.linear(outputs)
                logprobs = log_softmax(outputs, dim=1)
                #logprobs = calc_prob(outputs)
                logprobs, predicteds = logprobs.topk(beam_size, 1)

                for i in range(beam_size):
                    beam_candidates.append((b[0]+logprobs[:, i], b[1]+[predicteds[:, i]], hidden))

            beam_candidates.sort(reverse=True)
            beams = beam_candidates[:beam_size]

            nsteps += 1
            if nsteps >= max_step:
                break

        sampled_ids = []
        for id in beams[0][1]:
            sampled_ids.append(id)

        sampled_ids = torch.stack(sampled_ids)
        sampled_ids = torch.t(sampled_ids)

        return sampled_ids