# import torch


# class SAMMAV(torch.optim.Optimizer):

#     def __init__(self, params, params_mav, base_optimizer, rho=0.05, adaptive=True, **kwargs):
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

#         defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
#         super(SAMMAV, self).__init__(params, defaults)

#         self.mav_optimizer = torch.optim.Optimizer(params_mav, defaults)
#         self.param_groups_mav = self.mav_optimizer.param_groups
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups

#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group, group_mav in zip(self.param_groups, self.param_groups_mav):
#             scale = group["rho"] / (grad_norm + 1e-12)

#             for p, p_mav in zip(group["params"], group_mav["params"]):
#                 if p_mav.grad is None: continue
#                 self.state[p]["old_p"] = p.data.clone()
#                 e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p_mav.grad * scale.to(p)
#                 p.add_(e_w)  # climb to the local maximum "w + e(w)"

#         if zero_grad: self.mav_optimizer.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
#                 if "old_p" not in self.state[p]:
#                     print(p)
#                     print(list(self.state.keys()))
#                 else:
#                     p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

#         self.base_optimizer.step()  # do the actual "sharpness-aware" update

#         if zero_grad: self.zero_grad()

#     # @torch.no_grad()
#     # def step(self, closure=None):
#     #     assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
#     #     closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

#     #     self.first_step(zero_grad=True)
#     #     closure()
#     #     self.second_step()

#     def _grad_norm(self):
#         shared_device = self.param_groups_mav[0]["params"][
#             0].device  # put everything on the same device, in case of model parallelism
#         norm = torch.norm(
#             torch.stack(
#                 [
#                     ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2
#                                                                                 ).to(shared_device)
#                     for group in self.param_groups_mav
#                     for p in group["params"]
#                     if p.grad is not None
#                 ]
#             ),
#             p=2
#         )
#         return norm

#     def load_state_dict(self, state_dict):
#         super().load_state_dict(state_dict)
#         self.base_optimizer.param_groups = self.param_groups
