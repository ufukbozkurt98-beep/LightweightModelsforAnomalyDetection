"""
STLM core models:
  - Batch_Sam:       Fixed SAM ViT-H teacher (frozen at inference)
  - Batch_SamE:      Two-stream lightweight student (TinyViT × 2 + shared MaskDecoder)
  - SegmentationNet: Feature aggregation module (ResBlocks + ASPP)

Ported from https://github.com/Qi5Lei/STLM with minimal changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stlm_code.mobile_sam import sam_model_registry
from stlm_code.mobile_sam.modeling.tiny_vit_sam import TinyViT
from stlm_code.mobile_sam_decoder.mask_decoder import MaskDecoderSTLM
from stlm_code.model.model_utils import ASPP, BasicBlock, make_layer


def _replace_mask_decoder(model):
    """Replace the standard MaskDecoder with our STLM version that returns features."""
    original_decoder = model.mask_decoder
    new_decoder = MaskDecoderSTLM(
        transformer_dim=original_decoder.transformer_dim,
        transformer=original_decoder.transformer,
        num_multimask_outputs=original_decoder.num_multimask_outputs,
    )
    # Copy all matching weights from original decoder
    state = original_decoder.state_dict()
    new_decoder.load_state_dict(state, strict=True)
    model.mask_decoder = new_decoder
    return model


class Batch_Sam(nn.Module):
    """Fixed SAM teacher. Uses full SAM ViT-H as image encoder.
    Always in eval mode with frozen parameters during STLM training.
    """

    def __init__(self, sam_checkpoint, model_type, mode, device):
        super().__init__()
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.mode = mode
        self.device = device
        self.model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.model = _replace_mask_decoder(self.model)
        self.model = self.model.to(device=self.device)
        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image):
        transformed_image = image.permute(0, 3, 1, 2).contiguous()
        input_image = self.model.preprocess(transformed_image)
        pred_fea1 = []
        pred_fea2 = []
        if self.mode == "train":
            image_embeddings = self.model.image_encoder(input_image)
            for embedding in image_embeddings:
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                feature1, feature2 = self.model.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_fea1.append(feature1.squeeze(0))
                pred_fea2.append(feature2.squeeze(0))
        else:
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(input_image)
                for embedding in image_embeddings:
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    feature1, feature2 = self.model.mask_decoder(
                        image_embeddings=embedding.unsqueeze(0),
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_fea1.append(feature1.squeeze(0))
                    pred_fea2.append(feature2.squeeze(0))

        feature1 = torch.stack(pred_fea1, 0)
        feature2 = torch.stack(pred_fea2, 0)
        return feature1, feature2


class Batch_SamE(nn.Module):
    """Two-stream lightweight student model.
    - encoderT: plain student stream (TinyViT)
    - encoderS: denoising student stream (TinyViT)
    - Shared MaskDecoder from MobileSAM
    """

    def __init__(self, sam_checkpoint, model_type, mode, device):
        super().__init__()
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.mode = mode
        self.device = device
        self.model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.model = _replace_mask_decoder(self.model)
        self.model = self.model.to(device=self.device)
        self.encoderT = TinyViT(
            img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=True,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ).to(device=self.device)
        self.encoderS = TinyViT(
            img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=True,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ).to(device=self.device)
        if self.mode == "train":
            self.model.train()
            self.encoderT.train()
            self.encoderS.train()
        else:
            self.model.eval()
            self.encoderT.eval()
            self.encoderS.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image):
        transformed_image = image.permute(0, 3, 1, 2).contiguous()
        input_image = self.model.preprocess(transformed_image)

        pred_Tfea1 = []
        pred_Tfea2 = []
        pred_Sfea1 = []
        pred_Sfea2 = []
        if self.mode == "train":
            embeddingsT = self.encoderT(input_image)
            embeddingsS = self.encoderS(input_image)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            for embedding in embeddingsT:
                feature1, feature2 = self.model.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_Tfea1.append(feature1.squeeze(0))
                pred_Tfea2.append(feature2.squeeze(0))

            for embedding in embeddingsS:
                feature1, feature2 = self.model.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_Sfea1.append(feature1.squeeze(0))
                pred_Sfea2.append(feature2.squeeze(0))
        else:
            with torch.no_grad():
                embeddingsT = self.encoderT(input_image)
                embeddingsS = self.encoderS(input_image)

                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                for embedding in embeddingsT:
                    feature1, feature2 = self.model.mask_decoder(
                        image_embeddings=embedding.unsqueeze(0),
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_Tfea1.append(feature1.squeeze(0))
                    pred_Tfea2.append(feature2.squeeze(0))

                for embedding in embeddingsS:
                    feature1, feature2 = self.model.mask_decoder(
                        image_embeddings=embedding.unsqueeze(0),
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    pred_Sfea1.append(feature1.squeeze(0))
                    pred_Sfea2.append(feature2.squeeze(0))

        Tfeature1 = torch.stack(pred_Tfea1, 0)
        Tfeature2 = torch.stack(pred_Tfea2, 0)
        Sfeature1 = torch.stack(pred_Sfea1, 0)
        Sfeature2 = torch.stack(pred_Sfea2, 0)
        return [Tfeature1, Tfeature2], [Sfeature1, Sfeature2]


class SegmentationNet(nn.Module):
    """Feature Aggregation module: 2 ResBlocks + ASPP head."""

    def __init__(self, inplanes=448):
        super().__init__()
        self.res = make_layer(BasicBlock, inplanes, 128, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(128, 128, [1, 3]),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x
