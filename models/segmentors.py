import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from models.modules import AEIM, MRFM


class EISNet(nn.Module):
    def __init__(self, ver_ev='mit_b0', ver_img='mit_b2', num_classes=6, aet_rep=True, num_channels_ev=3,
                 num_channels_img=3, pretrained_ev=True, pretrained_img=True, weight_path='', if_viz=False):
        super().__init__()
        self.aet_rep = aet_rep
        dim_ev = [32, 64, 160, 256] if ver_ev == 'mit_b0' else [64, 128, 320, 512]
        dim_img = [32, 64, 160, 256] if ver_img == 'mit_b0' else [64, 128, 320, 512]
        # Event-based representation
        if self.aet_rep:
            self.rep = AEIM(in_dim=1, out_dim=dim_ev[0])
            self.num_channels_ev = num_channels_ev * 2
        else:
            self.num_channels_ev = num_channels_ev
        # Select an encoder
        self.encoder_ev = self.backbone_selector(ver_ev)
        self.encoder_img = self.backbone_selector(ver_img)
        # Use a pre-trained encoder
        if pretrained_ev:
            self.encoder_ev = self.load_pretrained_weights(encoder=self.encoder_ev, ver=ver_ev, weight_path=weight_path)
        if pretrained_img:
            self.encoder_img = self.load_pretrained_weights(encoder=self.encoder_img, ver=ver_img, weight_path=weight_path)
        # Replace the first layer of the encoder with a channel-adaptive layer
        from models.encoder.segformer_encoder import OverlapPatchEmbed
        if num_channels_ev != 3:
            self.encoder_ev.patch_embed1 = OverlapPatchEmbed(img_size=224, patch_size=7, stride=4,
                                                             in_chans=num_channels_ev, embed_dim=dim_ev[0])
        if num_channels_img != 3:
            self.encoder_img.patch_embed1 = OverlapPatchEmbed(img_size=224, patch_size=7, stride=4,
                                                              in_chans=num_channels_img, embed_dim=dim_img[0])
        # Multi-stage fusion
        self.fuse1 = MRFM(dim=[dim_ev[0], dim_img[0]])
        self.fuse2 = MRFM(dim=[dim_ev[1], dim_img[1]])
        self.fuse3 = MRFM(dim=[dim_ev[2], dim_img[2]])
        self.fuse4 = MRFM(dim=[dim_ev[3], dim_img[3]])
        # Decoder
        from models.decoder.segformer_decoder import SegFormerHead
        self.decoder = SegFormerHead(in_channels=dim_img, num_classes=num_classes)

    def backbone_selector(self, ver):
        assert ver in ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5']
        if ver == 'mit_b0':
            from models.encoder.segformer_encoder import mit_b0 as encoder
        elif ver == 'mit_b1':
            from models.encoder.segformer_encoder import mit_b1 as encoder
        elif ver == 'mit_b2':
            from models.encoder.segformer_encoder import mit_b2 as encoder
        elif ver == 'mit_b3':
            from models.encoder.segformer_encoder import mit_b3 as encoder
        elif ver == 'mit_b4':
            from models.encoder.segformer_encoder import mit_b4 as encoder
        elif ver == 'mit_b5':
            from models.encoder.segformer_encoder import mit_b5 as encoder
        else:
            from models.encoder.segformer_encoder import mit_b2 as encoder
        return encoder()

    def load_pretrained_weights(self, encoder, ver, weight_path):
        weights = torch.load(join(weight_path, ver + '.pth'), map_location='cpu')
        keys = []
        for k, v in weights.items():
            if k.startswith('head'):
                continue
            keys.append(k)
        weights = {k: weights[k] for k in keys}
        encoder.load_state_dict(weights)
        return encoder

    def forward(self, x_ev, x_img):
        B, _, H0, W0 = x_ev.shape
        outs = []
        """ Learn event-based representations """
        if self.aet_rep:
            x_ev = self.rep(x_ev[:, 0:self.num_channels_ev//2, :, :], x_ev[:, self.num_channels_ev//2:, :, :])
        else:
            x_ev = x_ev[:, 0:self.num_channels_ev, :, :]
        """ Encode the event-image fusion representation """
        # Stage 1
        # ==> Event branch
        x_ev, H, W = self.encoder_ev.patch_embed1(x_ev)
        for i, blk in enumerate(self.encoder_ev.block1):
            x_ev = blk(x_ev, H, W)
        x_ev = self.encoder_ev.norm1(x_ev)
        x_ev = x_ev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Image branch
        x_img, _, _ = self.encoder_img.patch_embed1(x_img)
        for i, blk in enumerate(self.encoder_img.block1):
            x_img = blk(x_img, H, W)
        x_img = self.encoder_img.norm1(x_img)
        x_img = x_img.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Fusion branch
        x, x_ev, x_img = self.fuse1(x_ev, x_img)
        outs.append(x)
        # Stage 2
        # ==> Event branch
        x_ev, H, W = self.encoder_ev.patch_embed2(x_ev)
        for i, blk in enumerate(self.encoder_ev.block2):
            x_ev = blk(x_ev, H, W)
        x_ev = self.encoder_ev.norm2(x_ev)
        x_ev = x_ev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Image branch
        x_img, H, W = self.encoder_img.patch_embed2(x_img)
        for i, blk in enumerate(self.encoder_img.block2):
            x_img = blk(x_img, H, W)
        x_img = self.encoder_img.norm2(x_img)
        x_img = x_img.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Fusion branch
        x, x_ev, x_img = self.fuse2(x_ev, x_img)
        outs.append(x)
        # Stage 3
        # ==> Event branch
        x_ev, H, W = self.encoder_ev.patch_embed3(x_ev)
        for i, blk in enumerate(self.encoder_ev.block3):
            x_ev = blk(x_ev, H, W)
        x_ev = self.encoder_ev.norm3(x_ev)
        x_ev = x_ev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Image branch
        x_img, _, _ = self.encoder_img.patch_embed3(x_img)
        for i, blk in enumerate(self.encoder_img.block3):
            x_img = blk(x_img, H, W)
        x_img = self.encoder_img.norm3(x_img)
        x_img = x_img.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Fusion branch
        x, x_ev, x_img = self.fuse3(x_ev, x_img)
        outs.append(x)
        # Stage 4
        # ==> Event branch
        x_ev, H, W = self.encoder_ev.patch_embed4(x_ev)
        for i, blk in enumerate(self.encoder_ev.block4):
            x_ev = blk(x_ev, H, W)
        x_ev = self.encoder_ev.norm4(x_ev)
        x_ev = x_ev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Image branch
        x_img, _, _ = self.encoder_img.patch_embed4(x_img)
        for i, blk in enumerate(self.encoder_img.block4):
            x_img = blk(x_img, H, W)
        x_img = self.encoder_img.norm4(x_img)
        x_img = x_img.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # ==> Fusion branch
        x, _, _ = self.fuse4(x_ev, x_img)
        outs.append(x)
        """ Generate the segmentation mask """
        x = self.decoder(outs)
        x = F.interpolate(x, size=[H0, W0], mode='bilinear', align_corners=False)
        return x
