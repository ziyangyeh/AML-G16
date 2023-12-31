import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, parse_shape, rearrange, reduce, repeat, unpack
from pointops import knn_query

from openpoints.models.backbone.pointnet import STN3d

from .build_models import MODELS
from .sub_models import AttentionFusion, GraphAttention, GVAPatchEmbed, PointTransformerV2Encoder, PointTransformerV2Layer, get_graph_feature_pair

@MODELS.register_module()
class TFuseNet(nn.Module):
    def __init__(
        self,
        img_enc="resnet34",
        img_enc_pretrained=True,
        k=8,
        img_channels=4,
        num_channels=24,
        dropout=0.5,
        num_classes=17,
        **kwargs,
    ) -> None:
        super(TFuseNet, self).__init__()
        self.k = k
        self.fstn_c = STN3d(num_channels // 2)
        self.fstn_n = STN3d(num_channels // 2)

        self.patch_embed = GVAPatchEmbed(
            in_channels=24,
            embed_channels=64,
            groups=8,
            depth=1,
            neighbours=8,
            pe_bias=False,
        )

        self.enc1 = PointTransformerV2Encoder(
            depth=2,
            in_channels=64,
            embed_channels=64,
            groups=16,
            neighbours=self.k,
            grid_size=0.06,
            qkv_bias=False,
            pe_bias=False,
        )

        self.enc2 = PointTransformerV2Encoder(
            depth=2,
            in_channels=64,
            embed_channels=128,
            groups=32,
            neighbours=self.k,
            grid_size=0.12,  # 0.06 0.12, 0.24, 0.48),
            qkv_bias=False,
            pe_bias=False,
        )

        self.enc3 = PointTransformerV2Encoder(
            depth=2,
            in_channels=128,
            embed_channels=256,
            groups=64,
            neighbours=self.k,
            grid_size=0.24,  # 0.06 0.12, 0.24, 0.48),
            qkv_bias=False,
            pe_bias=False,
        )

        self.conv1_c = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2_c = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3_c = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv1_n = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2_n = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3_n = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.graph_att1 = GraphAttention(feature_dim=12, out_dim=32, K=self.k)
        self.graph_att2 = GraphAttention(feature_dim=32, out_dim=64, K=self.k)
        self.graph_att3 = GraphAttention(feature_dim=64, out_dim=128, K=self.k)

        self.att_fusion1_1 = AttentionFusion(depth=0, dim=64, latent_dim=64, cross_dim_head=32, latent_dim_head=32)
        self.att_fusion2_1 = AttentionFusion(depth=0, dim=128, latent_dim=128, cross_dim_head=64, latent_dim_head=64)
        self.att_fusion3_1 = AttentionFusion(depth=0, dim=256, latent_dim=256, cross_dim_head=128, latent_dim_head=128)

        self.att_fusion1_2 = AttentionFusion(depth=0, dim=64, latent_dim=64, cross_dim_head=32, latent_dim_head=32)
        self.att_fusion2_2 = AttentionFusion(depth=0, dim=128, latent_dim=128, cross_dim_head=64, latent_dim_head=64)
        self.att_fusion3_2 = AttentionFusion(depth=0, dim=256, latent_dim=256, cross_dim_head=128, latent_dim_head=128)

        self.feature_extractor = timm.create_model(
            model_name=img_enc,
            pretrained=img_enc_pretrained,
            in_chans=img_channels,
            features_only=True,
        ).eval()

        self.image_conv2d1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.image_conv2d2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.image_conv2d3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.img_feat_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.coord_conv1d = nn.Sequential(
            nn.Conv1d(224, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.coord_pt = PointTransformerV2Layer(
            embed_channels=64,
            groups=8,
            qkv_bias=False,
            pe_bias=False,
        )
        self.norm_conv1d = nn.Sequential(
            nn.Conv1d(224, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.norm_pt = PointTransformerV2Layer(
            embed_channels=64,
            groups=8,
            qkv_bias=False,
            pe_bias=False,
        )

        self.att_fusion4 = AttentionFusion(depth=0, dim=448, latent_dim=448, cross_dim_head=224, latent_dim_head=224)

        self.pts_conv1d1 = nn.Sequential(
            nn.Conv1d(448, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pts_pt1 = PointTransformerV2Layer(
            embed_channels=256,
            groups=16,
            qkv_bias=False,
            pe_bias=False,
        )

        self.pts_conv1d2 = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pts_pt2 = PointTransformerV2Layer(
            embed_channels=128,
            groups=16,
            qkv_bias=False,
            pe_bias=False,
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, num_classes, 1, bias=False),
        )
    def forward(self, x, images):
        feat_shape = parse_shape(x, "b c n")

        # backbone - pre
        p0 = rearrange(x[:, :3, :], "b c n -> (b n) c").contiguous()
        x0 = rearrange(x, "b c n -> (b n) c").contiguous()
        o0 = torch.arange(1, feat_shape["b"] + 1, device=x.device, requires_grad=False) * feat_shape["n"]
        points = [p0, x0, o0]
        # GVA patch embed
        points = self.patch_embed(points)
        # stn coord&norm
        coord, norm = rearrange(x, "b (split c) n -> split b c n", split=2)

        trans_coord_feat = self.fstn_c(coord)
        coord = rearrange(coord, "b c n -> b n c")
        _feature = coord[:, :, 3:]
        coord = coord[:, :, :3]
        coord = torch.bmm(coord, trans_coord_feat)
        coord = torch.cat([coord, _feature], dim=2)
        coord = rearrange(coord, "b n c  -> b c n")

        trans_norm_feat = self.fstn_n(norm)
        norm = rearrange(norm, "b c n -> b n c")
        _feature = norm[:, :, 3:]
        norm = norm[:, :, :3]
        norm = torch.bmm(norm, trans_norm_feat)
        norm = torch.cat([norm, _feature], dim=2)
        norm = rearrange(norm, "b n c -> b c n")

        # image encoder
        img_shape = parse_shape(images, "b n c h w")
        batched_images = rearrange(images, "b n c h w -> (b n) c h w")
        batched_feats = self.feature_extractor(batched_images)
        last3feats = batched_feats[-3:]
        img_feats = [
            self.image_conv2d1(last3feats[0]),
            self.image_conv2d2(last3feats[1]),
            self.image_conv2d3(last3feats[2]),
        ]
        img_feats = [
            reduce(
                _feats,
                "(b n) c h w -> b h w c",
                "max",
                b=img_shape["b"],
                n=img_shape["n"],
            )
            for _feats in img_feats
        ]
        img_feats_skip = [self.flatten(self.img_feat_pool(rearrange(_feat, "b h w c -> b c h w"))) for _feat in img_feats]
        img_feats = [rearrange(_feats, "b h w c -> b (h w) c") for _feats in img_feats]

        # stage 1
        # graph module 1
        coord1, norm1, index = get_graph_feature_pair(coord, norm, k=self.k)
        coord1 = self.conv1_c(coord1)
        norm1 = self.conv1_n(norm1)
        coord1 = self.graph_att1(index, coord, coord1)
        norm1 = reduce(norm1, "b c n k -> b c n", "max")
        # backbone 1
        points1, cluster = self.enc1(points)
        points1[0] = points1[0][cluster]
        points1[1] = points1[1][cluster]
        points1[2] = torch.tensor(cluster.shape, device=points1[0].device)
        # att_fusion 1
        points1[1] = rearrange(
            self.att_fusion1_1(
                data=img_feats[0],
                queries_encoder=rearrange(points1[1], "(b n) c -> b n c", n=feat_shape["n"]),
            ),
            "b n c -> (b n) c",
        )

        # graph pt fusion 1
        cn1 = rearrange([coord1, norm1], "d b c n -> b n (d c)")
        points1[1] = rearrange(
            self.att_fusion1_2(
                data=cn1,
                queries_encoder=rearrange(points1[1], "(b n) c -> b n c", n=feat_shape["n"]),
            ),
            "b n c -> (b n) c",
        )

        # stage 2
        # graph module 2
        coord2, norm2, index = get_graph_feature_pair(coord1, norm1, k=self.k)
        coord2 = self.conv2_c(coord2)
        norm2 = self.conv2_n(norm2)
        coord2 = self.graph_att2(index, coord1, coord2)
        norm2 = reduce(norm2, "b c n k -> b c n", "max")
        # backbone 2
        points2, cluster = self.enc2(points1)
        points2[0] = points2[0][cluster]
        points2[1] = points2[1][cluster]
        points2[2] = torch.tensor(cluster.shape, device=points2[0].device)
        # att_fusion 2
        points2[1] = rearrange(
            self.att_fusion2_1(
                data=img_feats[1],
                queries_encoder=rearrange(points2[1], "(b n) c -> b n c", n=feat_shape["n"]),
            ),
            "b n c -> (b n)c",
        )
        # graph pt fusion 2
        cn2 = rearrange([coord2, norm2], "d b c n -> b n (d c)")
        points2[1] = rearrange(
            self.att_fusion2_2(
                data=cn2,
                queries_encoder=rearrange(points2[1], "(b n) c -> b n c", n=feat_shape["n"]),
            ),
            "b n c -> (b n) c",
        )

        # stage 3
        # graph module 3
        coord3, norm3, index = get_graph_feature_pair(coord2, norm2, k=self.k)
        coord3 = self.conv3_c(coord3)
        norm3 = self.conv3_n(norm3)
        coord3 = self.graph_att3(index, coord2, coord3)
        norm3 = reduce(norm3, "b c n k -> b c n", "max")
        # backbone 3
        points3, cluster = self.enc3(points2)
        points3[0] = points3[0][cluster]
        points3[1] = points3[1][cluster]
        points3[2] = torch.tensor(cluster.shape, device=points3[0].device)
        # att_fusion 3
        points3[1] = rearrange(
            self.att_fusion3_1(
                data=img_feats[2],
                queries_encoder=rearrange(points3[1], "(b n) c -> b n c", n=feat_shape["n"]),
            ),
            "b n c -> (b n)c",
        )
        # graph pt fusion 3
        cn3 = rearrange([coord3, norm3], "d b c n -> b n (d c)")
        points3[1] = rearrange(
            self.att_fusion3_2(
                data=cn3,
                queries_encoder=rearrange(points3[1], "(b n) c -> b n c", n=feat_shape["n"]),
            ),
            "b n c -> (b n) c",
        )

        # Segmentation
        coord, _ = pack([coord3, coord2, coord1], "b * n")
        norm, _ = pack([norm3, norm2, norm1], "b * n")

        img_feats = pack(img_feats_skip, "b *")[0]
        img_feats = repeat(img_feats, "b c -> b c n", n=feat_shape["n"])
        cn4, cn4_index = pack([coord, norm], "b * n")
        cn4 = self.att_fusion4(
            data=rearrange(img_feats, "b c n -> b n c"),
            queries_encoder=rearrange(cn4, "b c n -> b n c"),
        )
        cn4 = rearrange(cn4, "b n c -> b c n")
        coord, norm = unpack(cn4, cn4_index, "b * n")

        coord = self.coord_conv1d(coord)
        coord_shape = parse_shape(coord, "b c n")
        coord = [
            points3[0],
            rearrange(coord, "b c n -> (b n) c"),
            torch.tensor([coord_shape["n"] * coord_shape["b"]], device=coord.device).int(),
        ]
        reference_index, _ = knn_query(self.k, coord[0], coord[-1])
        coord = rearrange(
            self.coord_pt(coord, reference_index)[1],
            "(b n) c -> b c n",
            n=coord_shape["n"],
            b=coord_shape["b"],
        )

        norm = self.norm_conv1d(norm)
        norm_shape = parse_shape(norm, "b c n")
        norm = [
            points3[0],
            rearrange(norm, "b c n -> (b n) c"),
            torch.tensor([norm_shape["n"] * norm_shape["b"]], device=norm.device).int(),
        ]
        reference_index, _ = knn_query(self.k, norm[0], norm[-1])
        norm = rearrange(
            self.norm_pt(norm, reference_index)[1],
            "(b n) c -> b c n",
            n=norm_shape["n"],
            b=norm_shape["b"],
        )

        points_feats = rearrange(
            pack([points3[1], points2[1], points1[1]], "b *")[0],
            "(b n) c -> b c n",
            b=feat_shape["b"],
            n=feat_shape["n"],
        )

        points_feats = self.pts_conv1d1(points_feats)
        feats_shape = parse_shape(points_feats, "b c n")
        points_feats = [
            points3[0],
            rearrange(points_feats, "b c n -> (b n) c"),
            torch.tensor([feats_shape["n"] * feats_shape["b"]], device=points_feats.device).int(),
        ]
        reference_index, _ = knn_query(self.k, points_feats[0], points_feats[-1])
        points_feats = rearrange(
            self.pts_pt1(points_feats, reference_index)[1],
            "(b n) c -> b c n",
            n=feats_shape["n"],
            b=feats_shape["b"],
        )

        points_feats = self.pts_conv1d2(points_feats)
        feats_shape = parse_shape(points_feats, "b c n")
        points_feats = [
            points3[0],
            rearrange(points_feats, "b c n -> (b n) c"),
            torch.tensor([feats_shape["n"] * feats_shape["b"]], device=points_feats.device).int(),
        ]
        reference_index, _ = knn_query(self.k, points_feats[0], points_feats[-1])
        points_feats = rearrange(
            self.pts_pt2(points_feats, reference_index)[1],
            "(b n) c -> b c n",
            n=feats_shape["n"],
            b=feats_shape["b"],
        )

        feats = pack([points_feats, coord, norm], "b * n")[0]
        x = self.seg_head(feats)
        x = x.transpose(2, 1)
        return x
