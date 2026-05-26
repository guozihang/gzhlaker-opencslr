import torch.nn as nn

from ..modules.decoder import Decode


class sen_Decoder(nn.Module):
    def __init__(self, args, gloss_dict):
        super(sen_Decoder, self).__init__()
        search_mode = args.get("decode_mode", "beam")
        self.decoder = Decode(gloss_dict, args["num_classes"], search_mode)

    def forward(self, data):
        pred = None
        if not self.training:
            pred = self.decoder.decode(data["sequence_logits"], data["feat_len"], batch_first=False, probs=False)
        return {
            "recognized_sents": pred
        }
