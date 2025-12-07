import torch
import torch.nn.functional as F
import numpy as np


def grad_rollout(attentions, gradients, discard_ratio):
    print(f"[INFO] Number of attention layers: {len(attentions)}")
    print(f"[INFO] Number gradients: {len(gradients)}")

    if not attentions:
        return np.zeros((14, 14))

    # shoudl be CPU
    device = attentions[0].device
    if device != "cpu":
        print("[WARN] grad rollout not on cpu")
    result = torch.eye(attentions[0].size(-1)
                       ).to(device).unsqueeze(0)  # [1, T, T]

    with torch.no_grad():
        for i, (attention, grad) in enumerate(zip(attentions, gradients)):

            if grad is None:
                print(f"[ERROR] Layer {i} gradient is None. Using Ones.")
                weights = torch.ones_like(attention)
            else:
                weights = grad

            # Fuse Heads (Weighted Average)
            attention_heads_fused = (attention * weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Discard Ratio
            if discard_ratio > 0 and attention_heads_fused.max() > 0:
                flat = attention_heads_fused.view(
                    attention_heads_fused.size(0), -1)
                k = int(flat.size(-1) * discard_ratio)
                if k > 0:
                    _, indices = flat.topk(k, -1, largest=False)
                    flat[0, indices] = 0

            # Normalize
            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0 * I) / 2

            row_sums = a.sum(dim=-1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            a = a / row_sums

            result = torch.matmul(a, result)

    # Extract mask [Batch 0, Class Token 0, Image Patches 1:]
    mask = result[0, 0, 1:]

    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).cpu().numpy()

    m_min, m_max = mask.min(), mask.max()
    print(f"[INFO] Final Mask Range: {m_min:.4f} - {m_max:.4f}")

    if m_max - m_min > 1e-7:
        mask = (mask - m_min) / (m_max - m_min)
    else:
        mask = np.zeros_like(mask)

    return mask


class VITAttentionGradRollout:
    def __init__(self, model, device, attention_layer_name='attn_drop', discard_ratio=0.9):
        self.model = model
        self.original_device = device
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.attention_gradients = []
        self.hooks = []

        # 1. Apply Manual Patch to all MultiheadAttention layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                print(f"[INIT] Patching Attention Layer: {name}")

                # Replace the C++ forward with our Python forward
                module.forward = self._manual_attention_forward(module)

                # Register hook to capture the manually calculated weights
                self.hooks.append(module.register_forward_hook(
                    self._forward_hook_for_mha))

    def _manual_attention_forward(self, module):
        def forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
            is_batch_first = getattr(module, 'batch_first', False)

            if is_batch_first:
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)

            tgt_len, bsz, embed_dim = query.size()
            head_dim = embed_dim // module.num_heads

            q, k, v = F.linear(query, module.in_proj_weight,
                               module.in_proj_bias).chunk(3, dim=-1)

            # 2. Reshape for heads: (Seq, Batch, Heads, HeadDim) -> (Batch, Heads, Seq, HeadDim)
            q = q.contiguous().view(tgt_len, bsz, module.num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.contiguous().view(tgt_len, bsz, module.num_heads, head_dim).permute(1, 2, 0, 3)
            v = v.contiguous().view(tgt_len, bsz, module.num_heads, head_dim).permute(1, 2, 0, 3)

            attn_output_weights = torch.matmul(
                q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            # This matrix multiplication connects weights to output in the autograd graph
            attn_output = torch.matmul(attn_output_weights, v)

            # (Batch, Heads, Seq, HeadDim) -> (Seq, Batch, Heads, HeadDim) -> (Seq, Batch, Embed)
            attn_output = attn_output.permute(
                2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)

            attn_output = F.linear(
                attn_output, module.out_proj.weight, module.out_proj.bias)

            if is_batch_first:
                attn_output = attn_output.transpose(0, 1)

            # We force average_attn_weights=False here to return full heads
            return attn_output, attn_output_weights

        return forward

    def _forward_hook_for_mha(self, module, input, output):
        try:
            # Our patched forward returns (attn_output, attn_weights)
            attn_weights = output[1]
        except Exception:
            return

        # Save weights
        self.attentions.append(attn_weights.detach().cpu())

        idx = len(self.attention_gradients)
        self.attention_gradients.append(None)

        def save_grad(grad):
            self.attention_gradients[idx] = grad.detach().cpu()

        # Now that we use manual math, this tensor SHOULD require grad
        if attn_weights.requires_grad:
            attn_weights.retain_grad()
            attn_weights.register_hook(save_grad)
        else:
            print(
                "[ERROR] Hook failed: attn_weights does not require grad. Patch didn't work.")

    def __call__(self, input_tensor, category_index):
        self.attentions = []
        self.attention_gradients = []

        print("[INFO] Moving model to CPU for Interpretation...")
        self.model.to("cpu")
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to("cpu").requires_grad_(True)

        try:
            output = self.model(input_tensor)

            category_mask = torch.zeros_like(output)
            category_mask[:, category_index] = 1.0
            loss = (output * category_mask).sum()
            loss.backward()

            for i, grad in enumerate(self.attention_gradients):
                if grad is not None:
                    pass
                else:
                    print(f"[WARN] Gradient {i} is None. Fallback to Ones.")
                    self.attention_gradients[i] = torch.ones_like(
                        self.attentions[i])

            return grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)

        finally:
            print(f"[INFO] Restoring model to {self.original_device}...")
            self.model.to(self.original_device)

    def release(self):
        for hook in self.hooks:
            hook.remove()
