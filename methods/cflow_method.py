# methods/cflow_method.py
import torch
import torch.nn.functional as F
import numpy as np

from methods.cflow_freia import freia_cflow_head, positionalencoding2d, get_logp


class CFlowMethod:
    def __init__(
        self,
        extractor,
        coupling_blocks=8,
        condition_vec=128,
        clamp_alpha=1.9,
        fiber_batch=2048,
        device="cuda",
    ):
        self.extractor = extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.device = device
        self.coupling_blocks = coupling_blocks
        self.condition_vec = condition_vec
        self.clamp_alpha = clamp_alpha
        self.N = int(fiber_batch)

        self.decoders = None  # ModuleList (l1,l2,l3)

    def _build_decoders(self, feats_dict):
        dims = [feats_dict[k].shape[1] for k in ["l1", "l2", "l3"]]
        self.decoders = torch.nn.ModuleList(
            [
                freia_cflow_head(
                    n_feat=d,
                    n_coupling_blocks=self.coupling_blocks,
                    clamp=self.clamp_alpha,
                    cond_dim=self.condition_vec,
                )
                for d in dims
            ]
        ).to(self.device)

    def fit(self, train_loader, epochs=10, lr=2e-4):
        # Build decoders from first batch
        b0 = next(iter(train_loader))
        x0 = b0["image"].to(self.device)
        with torch.no_grad():
            f0 = self.extractor(x0)
        self._build_decoders(f0)

        opt = torch.optim.Adam(self.decoders.parameters(), lr=lr)

        self.decoders.train()
        for ep in range(epochs):
            total_loss = 0.0
            step_count = 0

            for batch in train_loader:
                x = batch["image"].to(self.device)

                with torch.no_grad():
                    feats = self.extractor(x)

                # CFLOW-AD train.py ile aynı: per-fiber step ve -logsigmoid(log_prob)
                for li, key in enumerate(["l1", "l2", "l3"]):
                    e = feats[key]  # (B,C,H,W)
                    B, C, H, W = e.shape
                    S = H * W
                    E = B * S

                    p = positionalencoding2d(self.condition_vec, H, W).to(self.device)  # (P,H,W)
                    p = p.unsqueeze(0).repeat(B, 1, 1, 1)  # (B,P,H,W)
                    c_r = p.reshape(B, self.condition_vec, S).transpose(1, 2).reshape(E, self.condition_vec)

                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)

                    perm = torch.randperm(E, device=self.device)
                    decoder = self.decoders[li]

                    # per-fiber
                    for start in range(0, E, self.N):
                        idx = perm[start : start + self.N]
                        c_p = c_r[idx]
                        e_p = e_r[idx]

                        z, log_j = decoder(e_p, c=[c_p])
                        decoder_log_prob = get_logp(C, z, log_j)
                        log_prob = decoder_log_prob / C

                        loss = -F.logsigmoid(log_prob).mean()

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        total_loss += float(loss.item())
                        step_count += 1

            print(f"[ep {ep}] loss={total_loss / max(step_count,1):.4f}")

        self.decoders.eval()

    @torch.no_grad()
    def predict(self, loader):
        """
        CFLOW-AD train.py ile aynı post-process:
          - her level için log_prob (per-dim) topla
          - level bazında global max çıkar, exp ile [0,1] prob yap
          - upsample edip topla
          - invert: super_mask = score_map.max() - score_map
          - image score = max(super_mask)
        """
        if self.decoders is None:
            raise RuntimeError("Decoders not initialized. Call fit() first.")

        self.decoders.eval()

        # 1) dataset boyunca level level log_prob biriktir
        level_logprobs = {"l1": [], "l2": [], "l3": []}
        level_hw = {"l1": None, "l2": None, "l3": None}

        out_h = None
        out_w = None

        for batch in loader:
            x = batch["image"].to(self.device)
            if out_h is None:
                out_h, out_w = x.shape[-2], x.shape[-1]

            feats = self.extractor(x)

            for li, key in enumerate(["l1", "l2", "l3"]):
                e = feats[key]  # (B,C,H,W)
                B, C, H, W = e.shape
                S = H * W
                E = B * S

                if level_hw[key] is None:
                    level_hw[key] = (H, W)

                p = positionalencoding2d(self.condition_vec, H, W).to(self.device)
                p = p.unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, self.condition_vec, S).transpose(1, 2).reshape(E, self.condition_vec)
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)

                decoder = self.decoders[li]

                # testte perm yok (orijinal kod gibi sırayla)
                for start in range(0, E, self.N):
                    idx = torch.arange(start, min(start + self.N, E), device=self.device)
                    c_p = c_r[idx]
                    e_p = e_r[idx]

                    z, log_j = decoder(e_p, c=[c_p])
                    decoder_log_prob = get_logp(C, z, log_j)
                    log_prob = (decoder_log_prob / C).detach().cpu()  # (N,)
                    level_logprobs[key].append(log_prob)

        # concat -> (E_total,)
        for key in level_logprobs:
            level_logprobs[key] = torch.cat(level_logprobs[key], dim=0).double()

        # 2) CFLOW-AD: normalize to (-inf, 0] by subtracting global max, then exp to [0,1]
        level_probs_maps = {}
        for key in ["l1", "l2", "l3"]:
            H, W = level_hw[key]
            lp = level_logprobs[key]
            lp = lp - torch.max(lp)          # global max subtraction
            prob = torch.exp(lp).float()     # [0,1]

            # reshape to (N_images, H, W)
            # E_total = N_images * (H*W)
            S = H * W
            assert prob.numel() % S == 0, f"{key}: prob size not divisible by H*W"
            n_img = prob.numel() // S
            prob_map = prob.view(n_img, H, W)  # (N,H,W)

            # upsample to input resolution
            up = F.interpolate(
                prob_map.unsqueeze(1), size=(out_h, out_w),
                mode="bilinear", align_corners=True
            ).squeeze(1)  # (N,out_h,out_w)

            level_probs_maps[key] = up

        # 3) aggregate probs: score_map = sum_k Pk
        score_map = level_probs_maps["l1"] + level_probs_maps["l2"] + level_probs_maps["l3"]

        # 4) invert to anomaly scores: super_mask = global_max - score_map
        super_mask = score_map.max() - score_map  # (N,H,W)

        # image-level score: max over pixels
        scores = super_mask.view(super_mask.shape[0], -1).max(dim=1).values  # (N,)

        maps = super_mask.unsqueeze(1)  # (N,1,H,W)
        return scores.cpu(), maps.cpu()