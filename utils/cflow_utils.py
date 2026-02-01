"""
CFLOW Training and Testing Utilities.

This module contains the training and testing loop implementations
for CFLOW, including learning rate scheduling and meta-epoch logic.
"""
import math
import numpy as np
from tqdm import tqdm
import time
import torch
from methods.cflow_freia import positionalencoding2d, activation, get_logp
import torch.nn.functional as F

log_theta = torch.nn.LogSigmoid()


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr  # Initial learning rate
    if c.lr_cosine:  # Use cosine learning rate schedule
        eta_min = lr * (c.lr_decay_rate**3)  # Set the minimum learning rate
        # Cosine decay
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    # Use step learning rate decay
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))  # Count how many decay points we passed
        if steps > 0:  # If we passed at least one decay epoch
            lr = lr * (c.lr_decay_rate**steps)  # Decay lr
    # Update all parameter groups
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  # Set the new learning rate


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    # Do warmup only in the first warmup epochs
    if c.lr_warm and epoch < c.lr_warm_epochs:

        # Warmup progress based on epoch and batch
        p = (batch_id + epoch * total_batches) / (c.lr_warm_epochs * total_batches)

        # Linearly increase lr from warmup_from to warmup_to
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Update all parameter groups
    for param_group in optimizer.param_groups:
        lrate = param_group["lr"]  # Set the new learning rate

    return lrate


def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):

    P = c.condition_vec  # Positional encoding size
    L = c.pool_layers  # Pool layer names (kept for compatibility)

    decoders = [decoder.train() for decoder in decoders]  # Set decoders to train mode
    adjust_learning_rate(c, optimizer, epoch)   # Set lr for this epoch

    I = len(loader)  # Number of batches in the loader
    iterator = iter(loader)

    for sub_epoch in range(c.sub_epochs):  # Repeat the loader multiple times per meta-epoch
        train_loss = 0.0  # Sum of losses
        train_count = 0  # Number of samples used for loss average

        # Loop over batches
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
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print("Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps".format(epoch, mean_test_loss, fps))

    return height, width, image_list, test_dist, gt_label_list, gt_mask_list


def t2np(tensor):
    # Convert a torch tensor to a numpy array
    return tensor.cpu().data.numpy() if tensor is not None else None
