"""SlowFast 专用解码器。"""

from models.keys import Keys, require
from models.modules.decoder import Decode


class SlowFast_Decoder:
    """分别解码主路径和卷积路径输出。"""

    def __init__(self, args, gloss_dict):
        self.decoder = Decode(
            gloss_dict, args["num_classes"], args.get("decode_mode", "beam")
        )

    def __call__(self, data):
        require(
            data,
            Keys.OUTPUT_FIRST,
            Keys.CONV_LOGITS,
            Keys.FEAT_LEN,
            who="SlowFast_Decoder",
        )
        pred = self.decoder.decode(
            data[Keys.OUTPUT_FIRST], data[Keys.FEAT_LEN], batch_first=False, probs=False
        )
        conv_pred = self.decoder.decode(
            data[Keys.CONV_LOGITS][0], data[Keys.FEAT_LEN], batch_first=False, probs=False
        )
        return {
            Keys.CONV_SENTS: conv_pred,
            Keys.RECOGNIZED_SENTS: pred,
        }
