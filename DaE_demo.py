import torch
import argparse

from Pipe_att_inf import Att_inference_pipe
from Pipe_cap_inf import Cap_inference_pipe


def DaE_Captioning(args):

    attribute = Att_inference_pipe(args.img_path, args.att_model_prefix)
    attribute = torch.Tensor(attribute)

    caption = Cap_inference_pipe(args.is_ko, args.img_path, attribute,
                                 args.cap_encoder_model_path, args.cap_decoder_model_path
                                 , args.data_file)

    print(caption)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--is_ko', type=int, default=1,
                        help='Korean: 1 / English: 0')

    parser.add_argument('--img_path', type=str, default='images/elephant.jpg',
                        help='image path')

    args = parser.parse_args()


    if args.is_ko:

        print('Korean Captioning progressing ...')

        parser.add_argument('--data_file', type=str, default='Pipe_files/ko/data-ko.p',
                            help='file path for saved ground truth attributes')

        parser.add_argument('--att_model_prefix', type=str, default='Pipe_files/ko/att-model-ko-',
                            help='weight file path for inference')

        parser.add_argument('--cap_encoder_model_path', type=str, default='Pipe_files/ko/encoder-model-ko.pkl',
                            help='trained attribute extraction model')

        parser.add_argument('--cap_decoder_model_path', type=str, default='Pipe_files/ko/decoder-model-ko.pkl',
                            help='trained attribute extraction model')

    else:

        print('English Captioning progressing ...')

        parser.add_argument('--data_file', type=str, default='Pipe_files/en/data.p',
                            help='file path for saved ground truth attributes')

        parser.add_argument('--att_model_prefix', type=str, default='Pipe_files/en/att-model-',
                            help='weight file path for inference')

        parser.add_argument('--cap_encoder_model_path', type=str, default='Pipe_files/en/encoder-model.pkl',
                            help='trained attribute extraction model')

        parser.add_argument('--cap_decoder_model_path', type=str, default='Pipe_files/en/decoder-model.pkl',
                            help='trained attribute extraction model')


    args = parser.parse_args()
    DaE_Captioning(args)