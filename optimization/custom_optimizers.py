"""
SAM Code from: https://github.com/davda54/sam/blob/main/sam.py

"""
import torch
import numpy as np

from utils import percentile


__all__ = ['SAM', 'TopkCrAM', 'NMTopkCrAM', 'TopkCrAMPeriod']


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def update_state_dict(self):
        # update the momentum buffer in the main optimizer state_dict
        # this is only to make sure the momentum is included in case of restarting from checkpoint
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p].update(self.base_optimizer.state[p])


class TopkCrAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, sparsities=[0.5, 0.7, 0.9], grad_norm=False, plus_version=True,
                 unif_prune=False, sparse_grad=True, **kwargs):
        super(TopkCrAM, self).__init__(params, base_optimizer, rho=rho, **kwargs)
        print("Using Topk CrAM")
        for group in self.param_groups:
            group['sparsities'] = sparsities
            group['grad_norm'] = grad_norm
            group['plus_version'] = plus_version
            group['unif_prune'] = unif_prune
            group['sparse_grad'] = sparse_grad
        print('==> base optimizer: ', self.base_optimizer)
        print('==> base optimizer defaults: ', self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = 1.
        if self.param_groups[0]["grad_norm"]:
            grad_norm = self._grad_norm() + 1e-12

        for group in self.param_groups:
            scale = group["rho"] / grad_norm
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if group["plus_version"]:
                    self.state[p]["clean_grad"] = p.grad.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        sparsities = self.param_groups[0]["sparsities"]
        k = np.random.choice(sparsities)

        if self.param_groups[0]["unif_prune"]:
            self._sparsify_weights_unif(k)
        else:
            self._sparsify_weights(k)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                if group['sparse_grad'] and ('mask' in self.state[p].keys()):
                    p.grad.mul_(self.state[p]['mask'])
                p.data = self.state[p]["old_p"]  # get back to "w" from "C(w + e(w))"
                if group["plus_version"]:
                    p.grad.add_(self.state[p]["clean_grad"])
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        self.update_state_dict()

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def _sparsify_weights(self, k):
        for group in self.param_groups:
            params_stats = None
            for p in group["params"]:
                if (len(p.shape) > 1) and (p.grad is not None):
                    p_stats = p.data.abs().view(-1)
                    if params_stats is None:
                        params_stats = p_stats
                    else:
                        params_stats = torch.cat((params_stats, p_stats))
            if params_stats is None:
                continue
            threshold = percentile(params_stats, k)
            for p in group["params"]:
                if (len(p.shape) > 1) and (p.grad is not None):
                    self.state[p]['mask'] = (p.data.abs() > threshold).float()
                    p.mul_(self.state[p]['mask'])

    @torch.no_grad()
    def _sparsify_weights_unif(self, k):
        for group in self.param_groups:
            params = list(group["params"])
            for i in range(1, len(params) - 2):
                if (len(params[i].shape) > 1) and (params[i].grad is not None):
                    p_stats = params[i].data.abs().view(-1)
                    threshold = percentile(p_stats, k)
                    self.state[params[i]]['mask'] = (params[i].data.abs() > threshold).float()
                    params[i].mul_(self.state[params[i]]['mask'])

    def get_sparsity(self):
        for group in self.param_groups:
            total_zeros = 0.
            total_params = 0.
            for p in group["params"]:
                total_zeros_p = (p.data == 0.).float().sum().item()
                total_zeros += total_zeros_p
                total_params += p.data.numel()
            print('sparsity model: ', total_zeros / total_params)


class NMTopkCrAM(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, grad_norm=False, sparse_grad=True, plus_version=True, **kwargs):
        super(NMTopkCrAM, self).__init__(params, base_optimizer, rho=rho, **kwargs)
        print("Using N:M Topk CrAM")
        for group in self.param_groups:
            group['grad_norm'] = grad_norm
            group['plus_version'] = plus_version
            group['sparse_grad'] = sparse_grad
        print('==> base optimizer: ', self.base_optimizer)
        print('==> base optimizer defaults: ', self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = 1.
        if self.param_groups[0]["grad_norm"]:
            grad_norm = self._grad_norm() + 1e-12

        for group in self.param_groups:
            scale = group["rho"] / grad_norm
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                if group['plus_version']:
                    self.state[p]['clean_grad'] = p.grad.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        k = np.random.uniform()
        if k > 0.5:
            self._sparsify_weights(2, 4)
        else:
            self._sparsify_weights(4, 8)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]  # get back to "w" from "C(w + e(w))"
                if p.grad is None:
                    continue
                if group['sparse_grad'] and ('mask' in self.state[p].keys()):
                    p.grad.mul_(self.state[p]['mask'])
                if group['plus_version']:
                    p.grad.add_(self.state[p]['clean_grad'])

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        self.update_state_dict()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _sparsify_weights(self, N, M):
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                if len(p.shape) == 4:
                    length = p.data.numel()
                    group = int(length / M)
                    weight_temp = p.data.abs().permute(0, 2, 3, 1).reshape(group, M)
                    index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

                    mask = torch.ones(weight_temp.shape, device=weight_temp.device)
                    mask = mask.scatter_(dim=1, index=index, value=0).reshape(p.permute(0, 2, 3, 1).shape)
                    mask = mask.permute(0, 3, 1, 2)
                    self.state[p]['mask'] = mask
                    p.mul_(self.state[p]['mask'])
                elif len(p.shape) == 2:
                    length = p.data.numel()
                    group = int(length / M)
                    weight_temp = p.data.abs().reshape(group, M)
                    index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

                    mask = torch.ones(weight_temp.shape, device=weight_temp.device)
                    mask = mask.scatter_(dim=1, index=index, value=0).reshape(p.shape)
                    self.state[p]['mask'] = mask
                    p.mul_(self.state[p]['mask'])
                else:
                    continue


class TopkCrAMPeriod(SAM):
    def __init__(self, params, base_optimizer, rho=0.05, sparsities=[0.5], grad_norm=False, plus_version=True,
                 unif_prune=False, sparse_grad=True, mask_freq=1, **kwargs):
        super(TopkCrAMPeriod, self).__init__(params, base_optimizer, rho=rho, **kwargs)
        print("Using Topk CrAM with periodical mask updates")
        for group in self.param_groups:
            group['sparsities'] = sparsities
            group['grad_norm'] = grad_norm
            group['plus_version'] = plus_version
            group['unif_prune'] = unif_prune
            group['sparse_grad'] = sparse_grad
        self.mask_freq = mask_freq
        self.mask_steps = {}
        self.k = None
        for s in sparsities:
            self.mask_steps[s] = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = 1.
        if self.param_groups[0]["grad_norm"]:
            grad_norm = self._grad_norm() + 1e-12

        for group in self.param_groups:
            scale = group["rho"] / grad_norm
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if group["plus_version"]:
                    self.state[p]["clean_grad"] = p.grad.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        sparsities = self.param_groups[0]["sparsities"]
        self.k = np.random.choice(sparsities)
        self.maybe_compute_masks_and_sparsify()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group['sparse_grad'] and (f'mask-{self.k}' in self.state[p].keys()):
                    p.grad.mul_(self.state[p][f"mask-{self.k}"])
                if group["plus_version"]:
                    p.grad.add_(self.state[p]["clean_grad"])
                p.data = self.state[p]["old_p"]  # get back to "w" from "C(w + e(w))"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _sparsify_weights(self):
        for group in self.param_groups:
            threshold = -1.
            if self.mask_steps[self.k] % self.mask_freq == 0:
                params_stats = None
                for p in group["params"]:
                    if (len(p.shape) > 1) and (p.grad is not None):
                        p_stats = torch.abs(p.data).view(-1)
                        if params_stats is None:
                            params_stats = p_stats
                        else:
                            params_stats = torch.cat((params_stats, p_stats))
                if params_stats is not None:
                    threshold = percentile(params_stats, self.k)

            for p in group["params"]:
                if (len(p.shape) > 1) and (p.grad is not None):
                    if self.mask_steps[self.k] % self.mask_freq == 0:
                        mask = (torch.abs(p.data) > threshold).float()
                        self.state[p][f"mask-{self.k}"] = mask
                    p.mul_(self.state[p][f'mask-{self.k}'])

    @torch.no_grad()
    def _sparsify_weights_unif(self):
        for group in self.param_groups:
            params = list(group["params"])
            for i in range(1, len(params) - 2):
                if (len(params[i].shape) > 1) and (params[i].grad is not None):
                    if self.mask_steps[self.k] % self.mask_freq == 0:
                        p_stats = params[i].data.abs().view(-1)
                        threshold = percentile(p_stats, self.k)
                        mask = (params[i].data.abs() > threshold).float()
                        self.state[params[i]][f'mask-{self.k}'] = mask
                    params[i].mul_(self.state[params[i]][f'mask-{self.k}'])

    def maybe_compute_masks_and_sparsify(self):
        if self.mask_steps[self.k] % self.mask_freq == 0:
            print(f'[{self.k}] updating mask at iteration {self.mask_steps[self.k]}')
        if self.param_groups[0]["unif_prune"]:
            self._sparsify_weights_unif()
        else:
            self._sparsify_weights()
        self.mask_steps[self.k] += 1

    @torch.no_grad()
    def get_sparsity(self):
        for group in self.param_groups:
            total_zeros = 0.
            total_params = 0.
            for p in group["params"]:
                zeros_p = (p.data == 0.).float().sum().item()
                params_p = p.data.numel()
                total_zeros += zeros_p
                total_params += params_p
                print(f'layer sparsity: {p.shape}    {zeros_p/params_p}')
            print('sparsity model: ', total_zeros / total_params)
