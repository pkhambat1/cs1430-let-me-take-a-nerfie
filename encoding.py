import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

def get_encoder(encoding, d=3, 
                multires=6, 
                degree=4,
                L=16, F=2, b=2, N_min=16, log2_hashmap_size=19, N_max=2048,
                **kwargs):



    if encoding == 'None':
        return lambda x, **kwargs: x, d
    
    elif encoding == 'frequency':
        encoder = FreqEncoder(input_dim=d, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)

    elif encoding == 'sphere_harmonics':
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=d, degree=degree)

    elif encoding == 'hashgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(d=d, L=L, F=F, b=b, N_min=N_min, log2_hashmap_size=log2_hashmap_size, N_max=N_max, gridtype='hash')
    
    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(d=d, L=L, F=F, N_min=N_min, log2_hashmap_size=log2_hashmap_size, N_max=N_max, gridtype='tiled')
    
    elif encoding == 'ash':
        from ashencoder import AshEncoder
        encoder = AshEncoder(input_dim=d, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=N_max)

    else:
        raise NotImplementedError()

    return encoder, encoder.output_dim