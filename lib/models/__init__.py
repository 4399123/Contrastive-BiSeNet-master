

#bisenetv2
from .bisenetv2 import BiSeNetV2

#bisenetv1
from .bisenetv1_repvit_m2 import BiSeNetV1_RepVit_M2
from .bisenetv1_fastvit_sa24 import BiSeNetV1_FastVit_SA24
from .bisenetv1_fastvit_sa12 import BiSeNetV1_FastVit_SA12
from .bisenetv1_fastvit_t12 import BiSeNetV1_FastVit_T12
from .bisenetv1_inceptionnext_tiny import BiSeNetV1_IncetionNeXt_Tiny
from .bisenetv1_repghostnet_200 import BiSeNetV1_RepGhostNet_200
from .bisenetv1_mobileone_s4 import BiSeNetV1_MobileOne_S4
from .bisenetv1_resnet34 import BiSeNetV1_ResNet34
from .bisenetv1_tiny_vit_21m import BiSeNetV1_Tiny_Vit_21m
from .bisenetv1_next_vit_small import BiSeNetV1_Next_Vit_Small
from .bisenetv1_mobilenetv4_medium import BiSeNetV1_MobileNetV4_Medium
from .bisenetv1_hgnetv2_b4 import BiSeNetV1_HGNetV2_B4
from .bisenetv1_repghostnet_080 import BiSeNetV1_RepGhostNet_080
from .bisenetv1_caformer_m36 import BiSeNetV1_Caformer_M36
from .bisenetv1_hgnetv2_b5 import BiSeNetV1_HGNetV2_B5
from .bisenetv1_hgnetv2_b0 import BiSeNetV1_HGNetV2_B0
from .bisenetv1_caformer_s36 import BiSeNetV1_Caformer_S36
from .bisenetv1_haloregnetz_b import BiSeNetV1_HaloRegNetZ_B
from .bisenetv1_ghostnetv2_100 import BiSeNetV1_GhostNetV2_100
from .bisenetv1_semnasnet_100 import BiSeNetV1_SemNasNet_100
from .bisenetv1_repvit_m1 import BiSeNetV1_RepVit_M1
from .bisenetv1_efficientnet_b3 import BiSeNetV1_EfficientNet_B3
from .bisenetv1_efficientnet_lite3 import BiSeNetV1_EfficientNet_Lite3
from .bisenetv1_cspdarknet import BiSeNetV1_CSPDarkNet
from .bisenetv1_regnety_040 import BiSeNetV1_RegNetY_040
from .bisenetv1_hgnetv2_b6 import BiSeNetV1_HGNetV2_B6

#segformer
from .segformer_b0 import SegFormer_B0

#topformer
from .topformer_efficientnet_lite1 import TopFormer_Lite1
from .topformer_hgnetv2_b5 import TopFormer_HGNetV2_B5

#pidnet
from .pidnet_s import PIDNet_S
from .pidnet_m import PIDNet_M
from .pidnet_l import PIDNet_L
from .pidnet_s_transformer import PIDNet_S_Transformer

#segnext
from .segnext_tiny import SegNeXt_Tiny
from .segnext_small import SegNeXt_Small


model_factory = {
    'bisenetv2': BiSeNetV2,
    'bisenetv1_fastvit_sa24': BiSeNetV1_FastVit_SA24,
    'bisenetv1_fastvit_sa12': BiSeNetV1_FastVit_SA12,
    'bisenetv1_fastvit_t12': BiSeNetV1_FastVit_T12,
    'bisenetv1_inceptionnext_tiny': BiSeNetV1_IncetionNeXt_Tiny,
    'bisenetv1_mobileone_s4': BiSeNetV1_MobileOne_S4,
    'bisenetv1_repghostnet_200': BiSeNetV1_RepGhostNet_200,
    'segformer_b0': SegFormer_B0,
    'bisenetv1_resnet34': BiSeNetV1_ResNet34,
    'topformer_efficientnet_lite1': TopFormer_Lite1,
    'bisenetv1_tiny_vit_21m': BiSeNetV1_Tiny_Vit_21m,
    'bisenetv1_next_vit_small': BiSeNetV1_Next_Vit_Small,
    'bisenetv1_mobilenetv4_medium': BiSeNetV1_MobileNetV4_Medium,
    'bisenetv1_hgnetv2_b4': BiSeNetV1_HGNetV2_B4,
    'bisenetv1_repghostnet_080': BiSeNetV1_RepGhostNet_080,
    'bisenetv1_caformer_m36': BiSeNetV1_Caformer_M36,
    'bisenetv1_hgnetv2_b5': BiSeNetV1_HGNetV2_B5,
    'bisenetv1_repvit_m2': BiSeNetV1_RepVit_M2,
    'bisenetv1_hgnetv2_b0': BiSeNetV1_HGNetV2_B0,
    'bisenetv1_caformer_s36': BiSeNetV1_Caformer_S36,
    'bisenetv1_haloregnetz_b': BiSeNetV1_HaloRegNetZ_B,
    'bisenetv1_ghostnetv2_100': BiSeNetV1_GhostNetV2_100,
    'bisenetv1_semnasnet_100': BiSeNetV1_SemNasNet_100,
    'bisenetv1_repvit_m1': BiSeNetV1_RepVit_M1,
    'bisenetv1_efficientnet_b3': BiSeNetV1_EfficientNet_B3,
    'pidnet_s': PIDNet_S,
    'pidnet_m': PIDNet_M,
    'pidnet_l': PIDNet_L,
    'pidnet_s_transformer': PIDNet_S_Transformer,
    'segnext_tiny':SegNeXt_Tiny,
    'segnext_small':SegNeXt_Small,
    'bisenetv1_efficientnet_lite3': BiSeNetV1_EfficientNet_Lite3,
    'bisenetv1_cspdarknet': BiSeNetV1_CSPDarkNet,
    'bisenetv1_regnety_040': BiSeNetV1_RegNetY_040,
    'topformer_hgnetv2_b5': TopFormer_HGNetV2_B5,
    'bisenetv1_hgnetv2_b6': BiSeNetV1_HGNetV2_B6,
}
