import torch
import torch.nn as nn


class CLIPVisualBackbone(nn.Module):
    """Frozen-by-caller CLIP image tower (the CLIP text branch is discarded).

    Parameters live under ``visual.*`` (checkpoint key prefix
    ``vis_encoder.backbone.visual.*``).
    """

    def __init__(self, model_name, download_root=None):
        super().__init__()
        from ..sematics.clip import load

        clip_model, _ = load(model_name, device="cpu", jit=False, download_root=download_root)
        self.model_name = model_name
        self.visual = clip_model.visual

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def extract_levels_anchored(self, x, levels):
        """v10: like extract_levels, but also returns the bit-exact CLIP CLS
        embedding computed from the SAME forward pass.

        Standard CLIP encode_image is ``proj @ ln_post(final[:, 0])`` (see
        VisionTransformer.forward in model/sematics/clip/model.py). The
        extracted token maps above are pre-ln_post residual streams (768-d),
        so a CLS derived by pooling them would NOT match the embedding the
        R6b baseline scores with. Anchoring v10's epoch-0 output to that
        baseline requires both from one forward:

            cls = ln_post(x[:, 0, :]) @ proj        # [B, 512], bit-exact CLIP

        Returns {"tokens": [...(B, N, 768) maps...], "cls": (B, 512)}.
        """
        visual = self.visual
        dtype = self.dtype
        wanted = {int(level) for level in levels}
        x = x.type(dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.to(dtype)
        x = torch.cat(
            [
                class_embedding
                + torch.zeros(
                    x.shape[0],
                    1,
                    class_embedding.shape[-1],
                    dtype=dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )
        x = x + visual.positional_embedding.to(dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, mirroring VisionTransformer.forward
        outs = []
        for index, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if index in wanted:
                outs.append(x.permute(1, 0, 2).float())  # back to NLD [B, N, C]
        if len(outs) != len(wanted):
            depth = len(visual.transformer.resblocks)
            raise ValueError(
                f"Requested CLIP block levels {sorted(wanted)} but tower has "
                f"only {depth} blocks."
            )
        x = x.permute(1, 0, 2)  # LND -> NLD before taking the CLS token
        cls = visual.ln_post(x[:, 0, :])
        if visual.proj is not None:
            cls = cls @ visual.proj
        return {"tokens": outs, "cls": cls.float()}
