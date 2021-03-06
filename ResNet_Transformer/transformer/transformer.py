import torch.nn as nn
#from .video_frontend import visual_model

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder, visual_model):
        super(Transformer, self).__init__()
        self.visual_frontend = visual_model
        self.encoder = encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        padded_input = padded_input.unsqueeze(1)
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        #print('length is: ', length)
        input_lengths = [len(padded_input[i]) for i in range(batch)]
        
        #print(padded_input.size(), input_lengths)
        encoder_padded_outputs, attn_score = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        #print(encoder_padded_outputs.size())
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        #print(pred.size(), max(pred.size(1)))
        return pred, gold, attn_score
    
    #def recognize_no_LM(self, padded_input, char_list, args):
    def recognize(self, padded_input):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        padded_input = padded_input.unsqueeze(1)
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        
        input_lengths = [len(padded_input[i]) for i in range(batch)]
        
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax

        #pred = self.decoder.recognize_beam(encoder_padded_outputs, char_list, args)
        pred = self.decoder.recognize(encoder_padded_outputs)
        #print(pred.size(), max(pred.size(1)))
        return pred
    
    def recognize_LM(self, padded_input, char_list, args):
    #def recognize(self, padded_input):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        padded_input = padded_input.unsqueeze(1)
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        
        input_lengths = [len(padded_input[i]) for i in range(batch)]
        
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax

        pred = self.decoder.recognize_beam(encoder_padded_outputs, char_list, args)
        #pred = self.decoder.recognize_beam(encoder_padded_outputs)
        #print(pred.size(), max(pred.size(1)))
        return pred
