from torch import nn
from layers import Tokenizer
from layers import Transformer


class ViTLite(nn.Module):
    def __init__(self,
                 img_size=32,
                 embedding_dim=256,
                 n_input_channels=3,
                 patch_size=4,
                 *args, **kwargs):
        super(ViTLite, self).__init__()
        assert img_size % patch_size == 0, f"Image size ({img_size}) has to be" \
                                           f"divisible by patch size ({patch_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = Transformer(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout_rate=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def vit_lite_100_feature_extractor():
    model = ViTLite(
        num_layers=7,
        num_heads=4,
        mlp_ratio=2,
        embedding_dim=512,
        patch_size=4,
        num_classes=100
    )

    class fc:
        in_features = 512

    model.fc = fc
    model.classifier.fc = nn.Identity()
    return model
