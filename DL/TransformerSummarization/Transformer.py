from Encoder import Encoder
from Decoder import Decoder
from basic_layers import *

class Transformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, d_ff=1024,
                 blocks_count=4, heads_count=8, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self._emb = nn.Sequential(
            nn.Embedding(source_vocab_size, d_model),
            PositionalEncoding(d_model, dropout_rate)
        )

        self.d_model = d_model
        self.encoder = Encoder(source_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)
        self.decoder = Decoder(target_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)
        # self.generator = Generator(d_model, target_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_inputs, target_inputs, source_mask, target_mask):
        source_inputs = self._emb(source_inputs)
        encoder_output = self.encoder(source_inputs, source_mask)

        target_inputs = self._emb(target_inputs)
        return self.decoder(target_inputs, encoder_output, source_mask, target_mask)

    # def generate_summary(self, source_inputs, target_inputs, source_mask, target_mask):
    #     with torch.no_grad():
    #         outputs = self.forward(source_inputs, target_inputs, source_mask, target_mask)
    #         return self.generator(outputs)
