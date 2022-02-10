import torch
import torch.nn as nn

from domainbed.lib.diversity.standard_diversity import DiversityLoss
from domainbed.lib.diversity import ib_utils, ib_mlp


class MiLoss(nn.modules.loss._Loss):

    def __init__(self, hparams):
        nn.modules.loss._Loss.__init__(self)
        self.hparams = hparams

    def forward(self, loglikelihood):
        # tanh clamp
        loglikelihood = ib_utils.tanh_clip(
            x=loglikelihood, clip_val=self.hparams.get("clamping_value")
        )
        # go to ]0, inf]
        likelihood = torch.exp(loglikelihood)
        # E[log(p/(1-p))] Donsker Varadhan
        E_dv = torch.mean(torch.log(likelihood + 1e-6))

        # - torch.log(1 - likelihood - 1e-6))
        return E_dv


class IBDiversity(DiversityLoss):

    def __init__(self, hparams, features_size, num_classes, **kwargs):
        DiversityLoss.__init__(self, hparams)

        self.conditional = self.hparams["conditional_d"]
        self.ib_space = self.hparams["ib_space"]
        self.sampling_negative = self.hparams.get("sampling_negative", "")
        self.reparameterization_var = float(self.hparams["reparameterization_var"])
        self.discriminator_loss = nn.BCEWithLogitsLoss()
        self.adversarial_loss = MiLoss(hparams)
        self.num_classes = num_classes

        self._init_discriminator(num_classes if self.ib_space == "logits" else features_size)

    def _init_discriminator(self, input_size):
        if self.conditional:
            self.discriminator = ib_mlp.CondDiscMLP(input_size, self.num_classes, self.hparams)
            self.embedding_layers = nn.ModuleList(
                [
                    nn.Embedding(self.num_classes, input_size)
                    for _ in range(self.hparams["num_members"])
                ]
            )
        else:
            self.discriminator = ib_mlp.DiscMLP(input_size, self.hparams)

        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
        )

    def forward(self, features_per_member, logits_per_member, classes, **kwargs):
        ibstats_per_member = logits_per_member if self.ib_space == "logits" else features_per_member
        ibstats_per_domain = ibstats_per_member.transpose(0, 1)

        # ibstats_per_domain is num_domains, num_members, bsize, size
        # training disc
        out_d, label_d = self.handle_tensors(ibstats_per_domain, classes)
        loss_d = self.discriminator_loss(out_d, label_d)
        self.optimizer_d.zero_grad()
        loss_d.backward(retain_graph=True)
        self.optimizer_d.step()

        # another sampling
        out_d, _ = self.handle_tensors(ibstats_per_domain, classes, only_positive=True)
        loss_adv = self.adversarial_loss(out_d)

        return {
            "loss_div": loss_adv,
            "loss_d": loss_d,
        }

    def handle_tensors(self, batch_ib, classes, only_positive=False):
        pos_input_d, pos_classes = ib_utils.positive_couples(batch_ib, classes)
        pos_label_d = torch.ones((pos_input_d.shape[0], 1)).to("cuda")
        if not only_positive:
            if self.conditional:
                assert self.sampling_negative == ""
                neg_input_d, neg_classes = ib_utils.negative_couples_conditional(
                    batch_ib,
                    batch_classes=classes,
                    num_classes=self.num_classes,
                    embedding_layers=self.embedding_layers
                )
            elif "perdomain" in self.sampling_negative:
                neg_input_d, neg_classes = ib_utils.negative_couples_per_domain(
                    batch_ib, batch_classes=classes
                )
            else:
                assert self.sampling_negative == ""
                neg_input_d, neg_classes = ib_utils.negative_couples(
                    batch_ib, batch_classes=classes
                )
            input_d = torch.cat((pos_input_d, neg_input_d), dim=0)
            neg_label_d = torch.zeros((neg_input_d.shape[0], 1)).to("cuda")

            label_d = torch.cat((pos_label_d, neg_label_d), dim=0)
            classes = torch.cat((pos_classes, neg_classes), dim=0)
        else:
            input_d = pos_input_d
            label_d = pos_label_d
            classes = pos_classes

        if self.reparameterization_var:
            input_d = ib_utils.reparameterization_trick(
                input_d, enc_var=self.reparameterization_var
            )

        if self.conditional:
            out_d = self.discriminator(input_d, classes)
        else:
            out_d = self.discriminator(input_d)
        return out_d, label_d
