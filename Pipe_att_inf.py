import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable


def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def Att_inference_pipe(img_path, att_model_prefix):

    def extract_tag(img_path, att_model_path):

        # Load the model
        model = torch.load(att_model_path)
        model.eval()

        # Load the trained weights
        if torch.cuda.is_available():
            model.cuda()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        image = Image.open(img_path).convert('RGB')
        if transform is not None:
            image = transform(image)

        image = to_var(image)
        image = torch.unsqueeze(image, dim=0)

        resnet = models.resnet152(pretrained=True)
        resnet_module = list(resnet.children())[:-1]
        CNN_model = nn.Sequential(*resnet_module)
        CNN_model.cuda()

        img_feats = CNN_model(image)
        img_feats = img_feats.view(img_feats.size(0), -1)
        atts_preds = model(img_feats)
        tag_out = atts_preds.data.cpu().numpy()

        return tag_out

    num_en = 5
    for en_idx in range(num_en):
        if en_idx == 0:
            tag_out_sum = extract_tag(img_path, att_model_prefix+str(en_idx)+'.pkl')
        else:
            tag_out = extract_tag(img_path, att_model_prefix+str(en_idx)+'.pkl')
            tag_out_sum = tag_out_sum + tag_out

    tag_out_en = tag_out_sum / num_en

    return tag_out_en

    return 0

if __name__ == '__main__':

    print('')