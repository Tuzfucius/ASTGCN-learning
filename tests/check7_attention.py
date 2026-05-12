import torch

from astgcn.models.attention import TemporalAttention


def main():
    B = 4
    N = 307
    C = 3
    T = 12

    x = torch.randn(B, N, C, T)

    temporal_attention = TemporalAttention(
        num_nodes=N,
        in_channels=C,
        num_timesteps=T,
    )

    E = temporal_attention(x)

    print("E shape:", E.shape)
    print("row sum:", E[0, 0].sum())


if __name__ == "__main__":
    main()