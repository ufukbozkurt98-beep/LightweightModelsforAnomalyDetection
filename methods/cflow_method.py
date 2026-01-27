# methods/cflow_method.py
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from methods.cflow_freia import load_decoder_arch, positionalencoding2d, activation, get_logp, t2np
from utils.cflow_utils import adjust_learning_rate, warmup_learning_rate

gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()

# ---- FIX: generator yerine len() olan wrapper ----
class TupleLoader:
    """
    DataLoader'ı (veya iterable loader'ı) orijinal CFLOW'un beklediği (image,label,mask)
    formatına çevirir ama len() ve iter() davranışını korur.
    """

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, dict):
                image = batch["image"]
                label = batch.get("label", torch.zeros(image.size(0), dtype=torch.long))
                mask = batch.get("mask", torch.zeros(image.size(0), 1, image.size(2), image.size(3)))
                yield image, label, mask
            else:
                yield batch


# ---- timm feature extractor adapter: hook yerine activation doldurur ----
class TimmActivationEncoder(torch.nn.Module):
    """
    Orijinal CFLOW: encoder forward hook ile activation['layer0/1/2'] doldurur.
    Biz: extractor(x)->{'l1','l2','l3'} alıp activation'a map ediyoruz.
    """

    def __init__(self, extractor, pool_layers):
        super().__init__()
        self.extractor = extractor
        self.pool_layers = pool_layers  # ['layer0','layer1','layer2']

    @torch.no_grad()
    def forward(self, x):
        feats = self.extractor(x)  # dict: l1,l2,l3
        activation[self.pool_layers[0]] = feats["l1"].detach()
        activation[self.pool_layers[1]] = feats["l2"].detach()
        activation[self.pool_layers[2]] = feats["l3"].detach()
        return feats


# ---- ORIJINAL train.py fonksiyonları ----
def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i in range(I):
            lr = warmup_learning_rate(c, epoch, i + sub_epoch * I, I * c.sub_epochs, optimizer)

            try:
                image, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, _ = next(iterator)

            image = image.to(c.device)
            with torch.no_grad():
                _ = encoder(image)

            for l, layer in enumerate(pool_layers):
                if "vit" in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[..., 1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)
                else:
                    e = activation[layer].detach()

                B, C, H, W = e.size()
                S = H * W
                E = B * S

                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)
                perm = torch.randperm(E).to(c.device)

                decoder = decoders[l]

                FIB = E // N
                assert FIB > 0, "MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!"
                for f in range(FIB):
                    idx = torch.arange(f * N, (f + 1) * N)
                    c_p = c_r[perm[idx]]
                    e_p = e_r[perm[idx]]

                    if "cflow" in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C
                    loss = -log_theta(log_prob)

                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()

                    train_loss += t2np(loss.sum())
                    train_count += len(loss)

        mean_train_loss = train_loss / train_count
        if c.verbose:
            print(
                "Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}".format(
                    epoch, sub_epoch, mean_train_loss, lr
                )
            )


def test_meta_epoch(c, epoch, loader, encoder, decoders, pool_layers, N):
    if c.verbose:
        print("\nCompute loss and scores on test set:")

    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for _layer in pool_layers]
    test_loss = 0.0
    test_count = 0
    start = time.time()

    with torch.no_grad():
        for i, (image, label, mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            image = image.to(c.device)
            _ = encoder(image)

            for l, layer in enumerate(pool_layers):
                if "vit" in c.enc_arch:
                    e = activation[layer].transpose(1, 2)[..., 1:]
                    e_hw = int(np.sqrt(e.size(2)))
                    e = e.reshape(-1, e.size(1), e_hw, e_hw)
                else:
                    e = activation[layer]

                B, C, H, W = e.size()
                S = H * W
                E = B * S

                if i == 0:
                    height.append(H)
                    width.append(W)

                p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)

                m = F.interpolate(mask, size=(H, W), mode="nearest")
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)

                decoder = decoders[l]
                FIB = E // N + int(E % N > 0)
                for f in range(FIB):
                    if f < (FIB - 1):
                        idx = torch.arange(f * N, (f + 1) * N)
                    else:
                        idx = torch.arange(f * N, E)

                    c_p = c_r[idx]
                    e_p = e_r[idx]
                    m_p = m_r[idx] > 0.5

                    if "cflow" in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p])
                    else:
                        z, log_jac_det = decoder(e_p)

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C
                    loss = -log_theta(log_prob)

                    test_loss += t2np(loss.sum())
                    test_count += len(loss)
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()

    fps = (len(loader.loader.dataset) / (time.time() - start))
        # \ if hasattr(loader, "loader") and hasattr(loader.loader, "dataset") else 0.0
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print("Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps".format(epoch, mean_test_loss, fps))

    return height, width, image_list, test_dist, gt_label_list, gt_mask_list


class _C:
    """Argparse config objesi yerine aynı attribute isimleriyle basit container."""
    pass


class CFlowMethod:
    """
    Senin extractor (timm features dict) + orijinal CFLOW train/test akışı.
    """

    def __init__(
        self,
        extractor,
        device="cuda",
        enc_arch="timm",
        dec_arch="freia-cflow",
        pool_layers=3,
        coupling_blocks=8,
        condition_vec=128,
        clamp_alpha=1.9,
        lr=2e-4,
        meta_epochs=25,
        sub_epochs=8,
        N=256,
        input_size=256,
        verbose=True,
        hide_tqdm_bar=False,
    ):
        self.extractor = extractor.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        c = _C()
        c.enc_arch = enc_arch
        c.dec_arch = dec_arch
        c.pool_layers = pool_layers
        c.coupling_blocks = coupling_blocks
        c.condition_vec = condition_vec
        c.clamp_alpha = clamp_alpha
        c.lr = lr
        c.meta_epochs = meta_epochs
        c.sub_epochs = sub_epochs
        c.verbose = verbose
        c.hide_tqdm_bar = hide_tqdm_bar

        c.crp_size = (input_size, input_size)

        # ORIJINAL main.py LR ayarları (mantık)
        c.lr_decay_epochs = [i * c.meta_epochs // 100 for i in [50, 75, 90]]
        c.lr_decay_rate = 0.1
        c.lr_warm_epochs = 2
        c.lr_warm = True
        c.lr_cosine = True
        if c.lr_warm:
            c.lr_warmup_from = c.lr / 10.0
            if c.lr_cosine:
                eta_min = c.lr * (c.lr_decay_rate**3)
                c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)
                ) / 2
            else:
                c.lr_warmup_to = c.lr

        c.device = torch.device(device)

        self.c = c
        self.N = N

        self.pool_layers = ["layer" + str(i) for i in range(pool_layers)]
        self.encoder = TimmActivationEncoder(self.extractor, self.pool_layers).to(device).eval()

        self.decoders = None
        self.optimizer = None

    def _build(self, train_loader):
        # ilk batch ile channel dim'leri çıkar
        image, _, _ = next(iter(TupleLoader(train_loader)))
        image = image.to(self.c.device)
        with torch.no_grad():
            _ = self.encoder(image)

        pool_dims = [activation[layer].size(1) for layer in self.pool_layers]

        self.decoders = [load_decoder_arch(self.c, dim_in).to(self.c.device) for dim_in in pool_dims]
        params = []
        for d in self.decoders:
            params += list(d.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.c.lr)

    def fit(self, train_loader):
        if self.decoders is None:
            self._build(train_loader)

        train_loader_t = TupleLoader(train_loader)

        for epoch in range(self.c.meta_epochs):
            print("Train meta epoch: {}".format(epoch))
            train_meta_epoch(
                self.c,
                epoch,
                train_loader_t,
                self.encoder,
                self.decoders,
                self.optimizer,
                self.pool_layers,
                self.N,
            )

        self.decoders = [d.eval() for d in self.decoders]

    @torch.no_grad()
    def predict(self, test_loader):
        if self.decoders is None:
            raise RuntimeError("Call fit() first.")

        test_loader_t = TupleLoader(test_loader)

        height, width, _, test_dist, _, _ = test_meta_epoch(
            self.c, 0, test_loader_t, self.encoder, self.decoders, self.pool_layers, self.N
        )

        # ORIJINAL score map üretimi
        test_map = [list() for _ in self.pool_layers]
        for l, _p in enumerate(self.pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)
            test_norm -= torch.max(test_norm)
            test_prob = torch.exp(test_norm)
            test_mask = test_prob.reshape(-1, height[l], width[l])

            test_map[l] = (
                F.interpolate(
                    test_mask.unsqueeze(1),
                    size=self.c.crp_size,
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(1)
                .cpu()
                .numpy()
            )

        score_map = np.zeros_like(test_map[0])
        for l, _p in enumerate(self.pool_layers):
            score_map += test_map[l]

        score_mask = score_map
        super_mask = score_mask.max() - score_mask
        score_label = np.max(super_mask, axis=(1, 2))

        scores = torch.from_numpy(score_label.astype(np.float32))
        maps = torch.from_numpy(super_mask.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)
        return scores, maps
