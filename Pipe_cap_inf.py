import torch
import _pickle
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable


def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def Cap_inference_pipe(is_ko, img_path, att, encoder_model_path, decoder_model_path, data_file):

    x = _pickle.load(open(data_file, "rb"))
    ixtoword = x[4]
    del x

    # Load models & weights
    encoder = torch.load(encoder_model_path)
    decoder = torch.load(decoder_model_path)

    encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    decoder.eval()

    # If use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = Image.open(img_path).convert('RGB')
    if transform is not None:
        image = transform(image)

    image = to_var(image)
    att = to_var(att)
    image = torch.unsqueeze(image, dim=0)

    resnet = models.resnet152(pretrained=True)
    resnet_module = list(resnet.children())[:-1]
    CNN_model = nn.Sequential(*resnet_module)
    CNN_model.cuda()

    img_feats = CNN_model(image)
    img_feats = img_feats.view(img_feats.size(0), -1)

    feature = encoder(img_feats)
    sampled_ids = decoder.sample_beam(feature, att, 5)

    # Decode word_ids to words
    if is_ko:

        for one_sample in sampled_ids:
            sampled_caption = []
            for word_id in one_sample:

                word = ixtoword[int(word_id)]
                split_word_part = word.split('|')
                if word != '<start>':
                    if word == '<eos>':
                        break
                    if not (split_word_part[1] == 'Josa' or split_word_part[1] == 'Eomi' or split_word_part[
                        1] == 'Suffix'):
                        sampled_caption.append(' ')

                    sampled_caption.append(split_word_part[0])

            sentence = ''.join(sampled_caption)
    else:
        for one_sample in sampled_ids:
            sampled_caption = []
            for word_id in one_sample:
                word = ixtoword[int(word_id)]
                if word == '.':
                    break
                sampled_caption.append(word)
            sentence = ' '.join(sampled_caption)

    return sentence


if __name__ == '__main__':
    print('')