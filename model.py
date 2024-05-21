import torch.nn as nn
import torch
from transformers import AutoModel
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1, padding_idx=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # define embedding layers for encoding positions
        self.pos_encoding = nn.Embedding(max_len, d_model, padding_idx=padding_idx)
        
    def forward(self, x):
        device = x.device
        chunk_size, B, d_model = x.shape
        position_ids = torch.arange(0, chunk_size, dtype=torch.int).unsqueeze(1).to(device)
        position_enc = self.pos_encoding(position_ids) # (chunk_size, 1, d_model)
        position_enc = position_enc.expand(chunk_size, B, d_model)
        
        # Add positional encoding to the input token embeddings
        x = x + position_enc
        x = self.dropout(x)
        
        return x
    
class CodeBertModel(nn.Module):
    def __init__(self, 
                 max_seq_length: int = 512, 
                 chunk_size: int = 512, 
                 dim_feedforward: int = 768,
                 padding_idx: int = 0,
                 n_attn_head: int = 2,
                 model_ckpt: str = 'microsoft/unixcoder-base'
                 ):
        super().__init__()
        self.embedding_model = AutoModel.from_pretrained(model_ckpt, trust_remote_code=True)
        
        dict_config = self.embedding_model.config.to_dict()
        for sym in ['hidden_dim', 'embed_dim', 'hidden_size']:
            if sym in dict_config.keys():
                embed_dim = dict_config[sym]
                
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=n_attn_head,
                                                   dim_feedforward=dim_feedforward,
                                                   batch_first=False)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=2,
                                                         )
        
        self.positional_encoding = PositionalEncoding(max_len=max_seq_length, 
                                                      d_model=embed_dim, 
                                                      padding_idx=padding_idx)
        
        self.loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 3.0]),
                                             label_smoothing=0.2)
        
        self.ffn = nn.Sequential(nn.Dropout(p=0.1),
                                 nn.Linear(embed_dim, 2)
                                 )
        self.chunk_size = chunk_size

    def prepare_chunk(self, input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor,
                            labels=None):
        """
        Prepare inputs into chunks that self.embedding_model can process (length < context_length)
        Shape info:
        - input_ids: (B, L)
        - attention_mask: (B, L)
        """
        
        device = input_ids.device
        # calculate number of chunks
        num_chunk = input_ids.shape[-1] // self.chunk_size
        if input_ids.shape[-1] % self.chunk_size != 0:
            num_chunk += 1
            pad_len = self.chunk_size - (input_ids.shape[-1] % self.chunk_size)
        else: 
            pad_len = 0
        
        B = input_ids.shape[0]
        # get the model's pad_token_id
        pad_token_id = self.embedding_model.config.pad_token_id
        
        # create a pad & zero tensor, then append it to the input_ids & attention_mask tensor respectively
        pad_tensor = torch.Tensor([pad_token_id]).expand(input_ids.shape[0], pad_len).int().to(device)
        zero_tensor = torch.zeros(input_ids.shape[0], pad_len).int().to(device)
        padded_input_ids = torch.cat([input_ids, pad_tensor], dim = -1).T # (chunk_size * num_chunk, B)
        padded_attention_mask = torch.cat([attention_mask, zero_tensor], dim = -1).T # (chunk_size * num_chunk, B)
                                                         
        chunked_input_ids = padded_input_ids.reshape(num_chunk, self.chunk_size, B).permute(0, 2, 1) # (num_chunk, B, chunk_size)
        chunked_attention_mask = padded_attention_mask.reshape(num_chunk, self.chunk_size, B).permute(0, 2, 1) # (num_chunk, B, chunk_size)
        
        pad_chunk_mask = self.create_chunk_key_padding_mask(chunked_input_ids)
        
        return chunked_input_ids, chunked_attention_mask, pad_chunk_mask
    
    def create_chunk_key_padding_mask(self, chunks):
        """
        If a chunk contains only pad tokens, ignore that chunk
        chunks: B, num_chunk, chunk_size
        """
        pad_token_id = self.embedding_model.config.pad_token_id
        pad_mask = (chunks == pad_token_id)
        
        num_pad = (torch.sum(pad_mask, -1) == self.chunk_size).permute(1, 0) # (num_chunk, B)
        
        return num_pad
    
    def forward(self, input_ids, attention_mask, labels=None):
        
        # calculate numbers of chunk
        chunked_input_ids, chunked_attention_mask, pad_chunk_mask = self.prepare_chunk(input_ids, attention_mask) # (num_chunk, B, chunk_size), (num_chunk, B, chunk_size), (num_chunk, B)
        
        # reshape input_ids & attention_mask tensors to fit into embedding model
        num_chunk, B, chunk_size = chunked_input_ids.shape
        chunked_input_ids, chunked_attention_mask = chunked_input_ids.contiguous().view(-1, chunk_size), chunked_attention_mask.contiguous().view(-1, self.chunk_size) # (B * num_chunk, chunk_size), (B * num_chunk, chunk_size)
        
        # embedded_chunks = (self.embedding_model(input_ids = chunked_input_ids,
        #                                         attention_mask = chunked_attention_mask) # (B * num_chunk, self.embedding_model.config.hidden_dim)
        #                        .view(num_chunk, B, -1) # (num_chunk, B, self.embedding_model.config.hidden_dim)
        #                   )
        
        embedded_chunks = (self.embedding_model(input_ids = chunked_input_ids,
                                        attention_mask = chunked_attention_mask)['pooler_output'] # (B * num_chunk, self.embedding_model.config.hidden_dim)
                        .view(num_chunk, B, -1) # (num_chunk, B, self.embedding_model.config.hidden_dim)
                    )
        
        # embedded_chunks = self.positional_encoding(embedded_chunks)
        
        # output = self.transformer_encoder(embedded_chunks, 
        #                                   src_key_padding_mask = pad_chunk_mask) # (num_chunk, B, self.embedding_model.config.hidden_dim)
        
        # logits = self.ffn(output[0])
        chunk_mean = torch.mean(embedded_chunks, dim=0) # ()
        logits = self.ffn(chunk_mean)
        if labels is not None:
            loss = self.loss_func(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}
    