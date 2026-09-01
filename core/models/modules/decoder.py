"""Decoder module for converting model outputs to text sequences.

Provides greedy max decoding and CTC beam search decoding for
converting sequence logits into recognized gloss sequences.
"""

import torch
from libs.ctcdecode import CTCBeamDecoder
from itertools import groupby
from models.keys import Keys, require


class Decode(object):
    """Core decoding logic with support for both greedy and beam search.

    Args:
        gloss_dict: Dictionary mapping gloss indices to (gloss_string, ...) tuples.
        num_classes: Number of gloss classes (including blank).
        search_mode: Decoding mode, either "max" (greedy) or "beam".
        blank_id: Index of the CTC blank token (default 0).
    """

    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                          num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """Decode model output into gloss sequences.

        Args:
            nn_output: Network output logits, shape (B, T, N) or (T, B, N).
            vid_lgt: Actual lengths of each sequence.
            batch_first: Whether batch dimension is first.
            probs: If True, nn_output is already probabilities.

        Returns:
            list: List of decoded sequences, each as [(gloss, idx), ...].
        """
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """CTC beam search decoding.

        Uses the built-in libs.ctcdecode CTCBeamDecoder (pure Python port) for beam search decoding.

        CTCBeamDecoder shapes:
            Input:  nn_output (B, T, N), should be passed through a softmax layer
            Output: beam_results (B, N_beams, T), int, decoded by i2g_dict
                    beam_scores (B, N_beams), p=1/np.exp(beam_score)
                    timesteps (B, N_beams)
                    out_lens (B, N_beams)

        Args:
            nn_output: Network output, shape (B, T, N).
            vid_lgt: Actual lengths of each sequence.
            probs: If True, nn_output is already probabilities.

        Returns:
            list: Decoded sequences with gloss labels and indices.
        """
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        """Greedy max decoding with CTC collapse.

        Takes argmax over vocabulary at each timestep, then collapses
        repeated labels and removes blank tokens.

        Args:
            nn_output: Network output, shape (B, T, N).
            vid_lgt: Actual lengths of each sequence.

        Returns:
            list: Decoded sequences with gloss labels and indices.
        """
        index_list = torch.argmax(nn_output, axis=2)
        vid_lgt = vid_lgt.cpu()
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:int(vid_lgt[batch_idx])])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list


class Decoder:
    """Wrapper for the Decode class used in the model pipeline.

    Integrates with the data dict flow: reads sequence logits and feature lengths
    from the data dict, decodes them, and returns recognized sentences.

    Args:
        args: Config dict containing "num_classes".
        gloss_dict: Dictionary mapping gloss indices to gloss strings.
    """

    def __init__ ( self , args, gloss_dict) :
        super ( Decoder , self ).__init__ ( )
        self.decoder = Decode ( gloss_dict , args["num_classes"] , args.get ( "decode_mode" , "beam" ) )
    def __call__ ( self , data) :
        """Decode model output from the data dict.

        Reads sequence logits and feature lengths, performs beam search decoding,
        and stores recognized sentences in the data dict.

        Args:
            data: Data dict containing Keys.SEQUENCE_LOGITS and Keys.FEAT_LEN.

        Returns:
            dict: Contains Keys.RECOGNIZED_SENTS with decoded gloss sequences.
        """
        require ( data , Keys.SEQUENCE_LOGITS , Keys.FEAT_LEN , who = "Decoder" )
        pred = self.decoder.decode ( data[Keys.SEQUENCE_LOGITS] , data[Keys.FEAT_LEN] , batch_first = False , probs = False )
        return {
            Keys.RECOGNIZED_SENTS: pred
        }