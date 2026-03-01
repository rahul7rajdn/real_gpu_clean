import math

import torch


class CustomFlashAttention(torch.nn.Module):
    """
    Custom FlashAttention implementation.

    Args:
        w_q: Query weight matrix of shape (hidden_dim, hidden_dim)
        w_k: Key weight matrix of shape (hidden_dim, hidden_dim)
        w_v: Value weight matrix of shape (hidden_dim, hidden_dim)
        w_o: Output weight matrix of shape (hidden_dim, hidden_dim)
        num_heads: Number of attention heads
    """

    def __init__(self, w_q, w_k, w_v, w_o, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_o = w_o

    def _flash_attention(self, q, k, v, T_r=128, T_c=128, causal=False):
        """
        FlashAttention implementation.

        Args:
            q: Query tensor of shape (batch_size, seq_len, hidden_dim)
            k: Key tensor of shape (batch_size, seq_len, hidden_dim)
            v: Value tensor of shape (batch_size, seq_len, hidden_dim)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """

        batch_size, seq_len, _ = q.shape

        # Split into heads: (B, H, N, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Output tensor in (B, H, N, D).
        o = torch.zeros_like(q)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Algorithm-1 style loop structure:
        # outer loop over (batch, head), then tiled KV loop, then tiled Q loop.
        for b in range(batch_size):
            for h in range(self.num_heads):
                q_bh = q[b, h]  # (N, D)
                k_bh = k[b, h]  # (N, D)
                v_bh = v[b, h]  # (N, D)

                # Streaming softmax stats for this head.
                m_bh = torch.full((seq_len,), float("-inf"), device=q.device, dtype=q.dtype)
                l_bh = torch.zeros((seq_len,), device=q.device, dtype=q.dtype)
                o_bh = torch.zeros((seq_len, self.head_dim), device=q.device, dtype=q.dtype)

                for c_start in range(0, seq_len, T_c):
                    c_end = min(c_start + T_c, seq_len)
                    k_j = k_bh[c_start:c_end, :]  # (Bc, D)
                    v_j = v_bh[c_start:c_end, :]  # (Bc, D)

                    if causal:
                        k_pos = torch.arange(c_start, c_end, device=q.device)

                    for r_start in range(0, seq_len, T_r):
                        r_end = min(r_start + T_r, seq_len)
                        q_i = q_bh[r_start:r_end, :]  # (Br, D)
                        o_i = o_bh[r_start:r_end, :]  # (Br, D)
                        m_i = m_bh[r_start:r_end]  # (Br,)
                        l_i = l_bh[r_start:r_end]  # (Br,)

                        # Score block S_ij = Q_i @ K_j^T / sqrt(d)
                        s_ij = (q_i @ k_j.transpose(-2, -1)) * scale  # (Br, Bc)

                        if causal:
                            q_pos = torch.arange(r_start, r_end, device=q.device)
                            causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                            s_ij = s_ij.masked_fill(causal_mask, float("-inf"))

                        # Blockwise softmax stats.
                        m_ij = s_ij.max(dim=-1).values  # (Br,)
                        m_ij_safe = torch.where(torch.isfinite(m_ij), m_ij, torch.zeros_like(m_ij))
                        p_ij = torch.exp(s_ij - m_ij_safe.unsqueeze(-1))
                        p_ij = torch.where(
                            torch.isfinite(m_ij).unsqueeze(-1),
                            p_ij,
                            torch.zeros_like(p_ij),
                        )
                        l_ij = p_ij.sum(dim=-1)  # (Br,)

                        # Online update:
                        # m_new = max(m_i, m_ij)
                        # l_new = exp(m_i-m_new)*l_i + exp(m_ij-m_new)*l_ij
                        # O_new = diag(l_new)^-1 * (diag(exp(m_i-m_new) * l_i) O_i
                        #         + diag(exp(m_ij-m_new)) (P_ij V_j))
                        m_new = torch.maximum(m_i, m_ij)
                        finite_m_new = torch.isfinite(m_new)
                        alpha = torch.where(
                            finite_m_new,
                            torch.exp(m_i - m_new),
                            torch.zeros_like(m_new),
                        )
                        beta = torch.where(
                            finite_m_new,
                            torch.exp(m_ij - m_new),
                            torch.zeros_like(m_new),
                        )
                        l_new = alpha * l_i + beta * l_ij

                        numer = (alpha * l_i).unsqueeze(-1) * o_i + beta.unsqueeze(-1) * (p_ij @ v_j)
                        o_new = torch.where(
                            (l_new > 0).unsqueeze(-1),
                            numer / l_new.unsqueeze(-1),
                            torch.zeros_like(numer),
                        )

                        o_bh[r_start:r_end, :] = o_new
                        m_bh[r_start:r_end] = m_new
                        l_bh[r_start:r_end] = l_new

                o[b, h] = o_bh

        # Reshape back to (batch_size, seq_len, hidden_dim)
        o = o.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)

        return o

    def forward(self, x, causal=False):
        """
        Forward pass for the self-attention layer using FlashAttention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            causal: Whether to use causal attention
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """

        # Obtain the query, key, and value tensors
        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        o = self._flash_attention(q, k, v, causal=causal)

        # Apply output projection
        o = o @ self.w_o.T
        return o
