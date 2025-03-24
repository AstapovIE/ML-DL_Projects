import torch
import torch.nn.functional as F


class Generator():
    def __init__(self, model, sos_token='<s>', eos_token='</s>', pad_token='<pad>'):
        super(Generator, self).__init__()
        self.model = model
        self.vocab = model.vocab
        self.ignore_tokens = {self.vocab[sos_token], self.vocab[eos_token], self.vocab[pad_token]}

    def decode_tensor(self, tensor):
        return ' '.join([self.vocab.itos[idx] for idx in tensor if idx not in self.ignore_tokens])

    def generate_summary(self, source_inputs, target_inputs, source_mask, target_mask):
        with torch.no_grad():
            outputs = self.model(source_inputs, target_inputs, source_mask, target_mask)
            return ([self.decode_tensor(seq) for seq in source_inputs],
                    [self.decode_tensor(seq)for seq in target_inputs],
                    [self.decode_tensor(F.log_softmax(seq, dim=-1).argmax(dim=-1)) for seq in outputs])
            # return ([self.decode_tensor(seq) for seq in source_inputs],
            #         [self.decode_tensor(seq).split("</s>")[0].strip()[4:] for seq in target_inputs],
            #         [self.decode_tensor(F.log_softmax(seq, dim=-1).argmax(dim=-1)).split("</s>")[0].strip() for seq in outputs])
