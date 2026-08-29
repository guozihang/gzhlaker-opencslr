import torch.nn as nn

from ..modules.decoder import Decode
from ..keys import Keys, require


class sen_Decoder(nn.Module):
    def __init__(self, args, gloss_dict):
        super(sen_Decoder, self).__init__()
        search_mode = args.get("decode_mode", "beam")
        self.decoder = Decode(gloss_dict, args["num_classes"], search_mode)

    def forward(self, data):
        require(data, Keys.SEQUENCE_LOGITS, Keys.FEAT_LEN, who="sen_Decoder")
        pred = None
        if not self.training:
            pred = self.decoder.decode(data[Keys.SEQUENCE_LOGITS], data[Keys.FEAT_LEN], batch_first=False, probs=False)
        return {
            Keys.RECOGNIZED_SENTS: pred
        }
