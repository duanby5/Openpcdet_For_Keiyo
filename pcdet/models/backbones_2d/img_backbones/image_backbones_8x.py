import torch
import torch.nn as nn

from pcdet.models.model_utils.basic_block_2d import BasicBlock2D
import copy

class ImageBackbones8X(nn.Module):

    def __init__(self, model_cfg):
        """
        Initializes IMAGE 2D convolution module
        Args:
            model_cfg: EasyDict, Model configuration
        """
        super().__init__()
        assert len(model_cfg.STRIDES) == len(model_cfg.OUTPUT_CHANNELS), \
            f'length of strides does not match the output list'

        self.input_channel = model_cfg.INPUT_CHANNEL
        self.strides = model_cfg.STRIDES
        self.output_channels = model_cfg.OUTPUT_CHANNELS
        self.input_channels = [self.input_channel]
        self.input_channels.extend(self.output_channels[:-1])
        self.blocks = nn.ModuleList()
        num_levels = len(self.strides)

        kwargs = {
            'kernel_size': 3,
            'padding': 1,
        }

        for idx in range(num_levels):
            cur_kwargs = copy.deepcopy(kwargs)
            cur_kwargs.update({'stride': self.strides[idx]})
            cur_layers = [BasicBlock2D(in_channels=self.input_channels[idx], out_channels=self.output_channels[idx], **cur_kwargs)]
            self.blocks.append(nn.Sequential(*cur_layers))

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images

        # Create a mask for padded pixels
        mask = torch.isnan(x)

        # Make padded pixels = 0
        x[mask] = 0

        return x

    def forward(self, batch_dict):
        """
        simply extract feature from image
        Args:
            batch_dict:
                image_features: (B, C, H, W)
        Returns:
            batch_dict:
                image_features_seq: dict type structure containing [origin, conv_2x, conv_4x, conv_8x]
        """

        image_features = batch_dict["images"]

        # Preprocess images
        image_features = self.preprocess(image_features)

        x = image_features
        multi_scale_image_strides = {'origin': 1}
        multi_scale_image_features = {'origin': x}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            multi_scale_image_features[f'conv_{2**(i+1)}x'] = x
            multi_scale_image_strides[f'conv_{2**(i+1)}x'] = 2**(i+1)

        batch_dict.update({
            'multi_scale_image_features': multi_scale_image_features,
            ' multi_scale_image_strides': multi_scale_image_strides,
        })

        return batch_dict

