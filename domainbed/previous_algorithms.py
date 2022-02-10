from domainbed.algorithms import *



class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = ["gaussian"]
        else:
            self.kernel_type = ["mean_cov"]

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(self.num_domains):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, self.num_domains):
                penalty += torchutils.mmd(features[i], features[j], self.kernel_type)

        objective /= self.num_domains
        if self.num_domains > 1:
            penalty /= self.num_domains * (self.num_domains - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_lambda"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)



class COREL(ERM):
    """
    Adaptative COREL
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(COREL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.register_buffer("update_count", torch.tensor([0]))
        self.ema_per_domain_mean = [
            misc.MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self.ema_per_domain_var = [
            misc.MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self.list_methods = self.hparams["method"].split("_")

    def update(self, minibatches, unlabeled=None):

        features = [self.featurizer(xi) for xi, _ in minibatches]
        logits = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]
        nlls = [F.cross_entropy(logits[i], targets[i]) for i in range(self.num_domains)]
        all_nll = torch.stack(nlls, dim=0).mean(dim=0)

        dict_penalty = self.compute_corel_penalty(features, logits, targets)
        penalty_active = float(self.update_count >= self.hparams["penalty_anneal_iters"])
        if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
            # gradient magnitudes that happens at this step.
            self._init_optimizer()
        self.update_count += 1
        penalty = (
            self.hparams["lambda"] * dict_penalty["penalty_var"] + self.hparams["lambdamean"] * dict_penalty["penalty_mean"])
        objective = all_nll + penalty_active * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        output_dict = {'loss': objective.item(), 'nll': all_nll.item(), "penalty": penalty.item()}
        output_dict.update({key: value.item() for key, value in dict_penalty.items()})
        for i in range(self.num_domains):
            output_dict["nll_" + str(i)] = nlls[i].item()
        return output_dict

    def compute_corel_penalty(self, features, logits, targets):
        weights = [
            self._compute_weight_per_sample(logits[i], targets[i]) for i in range(self.num_domains)
        ]
        feats_mean_per_domain, feats_var_per_domain = self._compute_feats(features, weights)
        penalty_mean = self._compute_distance(feats_mean_per_domain)
        penalty_var = self._compute_distance(feats_var_per_domain)
        return {"penalty_var": penalty_var, "penalty_mean": penalty_mean}

    def _compute_weight_per_sample(self, logits, targets):
        if "mse" in self.list_methods:
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
            return nn.MSELoss(reduction="none")(torch.softmax(logits, dim=1), one_hot_targets).mean(dim=1)
        return nn.CrossEntropyLoss(reduction='none')(logits, targets)

    def _compute_feats(self, features, weights):
        feats_var_per_domain = []
        feats_mean_per_domain = []
        for i in range(self.num_domains):
            x = features[i]
            weights_x = weights[i]

            mean_x = x.mean(0, keepdim=True)
            feats_mean_per_domain.append(mean_x)

            transf_x = x
            if "notcentered" not in self.list_methods:
                transf_x = transf_x - mean_x

            if "weight" in self.list_methods:
                transf_x = transf_x * weights_x.view((weights_x.size(0), 1))

            if "offdiagonal" in self.list_methods:
                cova_x = torch.einsum("na,nb->ab", transf_x,
                                      transf_x) / (transf_x.size(0) * transf_x.size(1))
            else:
                cova_x = (transf_x).pow(2).mean(dim=0)
            feats_var_per_domain.append(cova_x)

        for domain_id in range(self.num_domains):
            feats_mean_per_domain[domain_id] = self.ema_per_domain_mean[domain_id].update_value(
                feats_mean_per_domain[domain_id]
            )
            feats_var_per_domain[domain_id] = self.ema_per_domain_var[domain_id].update_value(
                feats_var_per_domain[domain_id]
            )

        return feats_mean_per_domain, feats_var_per_domain

    def _compute_distance(self, feats_per_domain):
        # compute gradient variances averaged across domains
        feats_mean = torch.stack(feats_per_domain, dim=0).mean(dim=0)

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += (feats_per_domain[domain_id] - feats_mean).pow(2).mean()
        return penalty / self.num_domains


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_optimizer = torch.optim.Adam(
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1_d"], 0.9),
        )

        self.optimizer = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1_d"], 0.9),
        )

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_logits = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_logits, all_y)
            objective = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_optimizer.zero_grad()
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()
            return {"objective": objective.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )




class IRMAdv(Algorithm):
    """Invariant Risk Minimization with adversarial training"""
    CUSTOM_FORWARD = True

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional=True):

        super(IRMAdv, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

        # Algorithms and classifiers
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams["nonlinear_classifier"],
                name="main"
            )
        )
        self.classifier_per_domain = nn.ModuleList(
            [
                extend(
                    networks.Classifier(
                        self.featurizer.n_outputs,
                        num_classes,
                        self.hparams["nonlinear_classifier"],
                        name=str(i)
                    )
                ) for i in range(num_domains)
            ]
        )

        ## Optimizer
        self._init_optimizer()

        # Domain classifier
        if self.hparams["lr_d"] != 0:
            self.conditional = conditional
            self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
            self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

            ## Optimizer
            self.disc_optimizer = torch.optim.Adam(
                list(self.discriminator.parameters()) + list(self.class_embeddings.parameters()) +
                list(self.classifier_per_domain.parameters()),
                lr=self.hparams["lr_d"],
                weight_decay=self.hparams["weight_decay_d"],
                betas=(self.hparams["beta1_d"], 0.9),
            )

        self.loss_backpack = extend(nn.CrossEntropyLoss(reduction='none'))

    def _init_optimizer(self):

        betas = ((self.hparams["beta1"], 0.9) if self.hparams["beta1"] != 0.9 else (0.9, 0.999))
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            betas=betas,
        )

    def zero_grad(self):
        if self.hparams["lr_d"] != 0:
            self.disc_optimizer.zero_grad()
        self.optimizer.zero_grad()

    def _call_classifier_per_domain(self, all_z, minibatches, permute=False):
        preds = []
        all_z_idx = 0
        for i, b in enumerate(minibatches):
            x = b["x"]
            z = all_z[all_z_idx:all_z_idx + x.shape[0]]
            all_z_idx += x.shape[0]
            if permute:
                i = (i + 1) % (len(minibatches))
            preds.append(self.classifier_per_domain[i](z))
        return torch.cat(preds)

    def _get_mmdfeatures_loss(self, all_z, minibatches, method, dict_residual=None):
        features = []
        all_z_idx = 0
        for i, b in enumerate(minibatches):
            x = b["x"]
            features.append(all_z[all_z_idx:all_z_idx + x.shape[0]])
            all_z_idx += x.shape[0]

        penalty = 0
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                penalty += torchutils.mmd(features[i], features[j], kernel_type=method)

        if self.num_domains > 1:
            penalty /= self.num_domains * (self.num_domains - 1) / 2

        dict_output = {"penalty_features": penalty}

        return dict_output, penalty, {}

    def _get_domaindisc_loss(self, all_z, all_y, minibatches):
        device = "cuda" if minibatches[0]["x"].is_cuda else "cpu"
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((b["x"].shape[0],), i, dtype=torch.int64, device=device)
                for i, b in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)
        return disc_loss

    def update(self, minibatches, unlabeled=None):
        assert self.num_domains == len(minibatches)
        self.update_count += 1
        all_x = torch.cat([b["x"] for b in minibatches])
        all_y = torch.cat([b["y"] for b in minibatches])
        all_z = self.featurizer(all_x)

        self.zero_grad()
        dict_output = {}

        # 1. domain discriminator loss
        if self.hparams["lr_d"] != 0:
            if self.hparams["disc_lambda"]:
                disc_loss_to_opt = self._get_domaindisc_loss(all_z.detach(), all_y, minibatches)
            else:
                disc_loss_to_opt = torch.tensor(0)

            loss_per_domain_to_opt = F.cross_entropy(
                self._call_classifier_per_domain(all_z.detach(), minibatches), all_y
            )
            (disc_loss_to_opt + loss_per_domain_to_opt).backward()
            self.disc_optimizer.step()
            dict_output.update({"l_disc": disc_loss_to_opt, "l_pd": loss_per_domain_to_opt})

        # 2. features/classifier losses

        strategy = self.hparams["strategy"]
        all_logits = self.classifier(all_z)

        ## 2.1 classification losses
        if strategy not in [61]:
            all_loss = F.cross_entropy(all_logits, all_y)
        else:
            # raise DeprecationWarning
            all_loss = F.cross_entropy(self._call_classifier_per_domain(all_z, minibatches), all_y)

        ## 2.1 compute adv_loss = soft_kl(all_logits, disc_out)
        dict_output_align, loss_align = self.call_alignment(
            strategy, all_z, all_logits, minibatches
        )
        dict_output.update(dict_output_align)

        ## 2.3 domain disc loss
        if self.hparams["disc_lambda"]:
            adv_disc_loss = -self._get_domaindisc_loss(all_z, all_y, minibatches)
            dict_output["l_disc_adv"] = adv_disc_loss
        else:
            adv_disc_loss = torch.tensor(0)

        ## 2.4 reg multiplier
        coeff_reg = FisherMMD.get_current_consistency_weight(self)

        if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            print("Finally reach penalty_anneal_iters: " + str(self.update_count))
            # self._init_optimizer()

        ## 2.5 final loss
        penalty = (loss_align + self.hparams["disc_lambda"] * adv_disc_loss)
        objective = all_loss + coeff_reg * penalty
        if self.hparams["penalty_method"] in [3, 4]:
            raise ValueError("penalty_method")
            objective /= 1 + coeff_reg * (
                self.hparams["mmd_lambda"] + self.hparams["mean_lambda"] +
                self.hparams["disc_lambda"]
            )

        # 2.6 backpropagate and logs
        self.zero_grad()
        objective.backward()
        self.optimizer.step()
        dict_output.update({
            "l": all_loss,
            "l_align": loss_align,
            "l_gen": objective,
        })

        return {k: v.item() for k, v in dict_output.items()}

    def get_tb_dict(self):
        dict_output = {
            "coeff_reg": FisherMMD.get_current_consistency_weight(self),
            "pd_lambda": self.hparams["pd_lambda"],
            "mmd_lambda": self.hparams["mmd_lambda"],
            "mean_lambda": self.hparams["mean_lambda"],
            "disc_lambda": self.hparams["disc_lambda"]
        }
        if self.hparams["strategy"] != "none":
            dict_output["strategy"] = self.hparams["strategy"]
        return dict_output

    def call_alignment(self, strategy, all_z, all_logits, minibatches):
        if strategy in ["none", 100, 61]:
            return {}, torch.tensor(0)

        dict_output = {}
        loss_align = 0

        if strategy in [-1, 0, 1, 2]:
            loss_align += self.call_alignment_method_logits(
                all_z, all_logits, minibatches, method=strategy
            )
            dict_output["l_align_logits"] = loss_align
            assert self.hparams["pd_lambda"] == 0
            return dict_output, self.hparams["lambda_mmd"] * loss_align

        DICT_STRATEGY_TO_METHOD = {
            3: "l2_detach",
            31: "l2",
            33: "l2_detach_filtered",
            51: "ntk_detach"
        }
        DICT_STRATEGY_TO_METHOD_FULL = {71: "ntk_detach", 72: "ntk_domain", 7180: "ntk_detach"}

        DICT_STRATEGY_TO_METHOD_FEATURES = {90: "linear", 91: "gaussian"}

        pd_reg = losses.get_current_consistency_weight(
            epoch=self.update_count.item(),
            consistency_rampup=self.hparams["pd_penalty_anneal_iters"],
            epoch_start_consistency_rampup=0,
            penalty_anneal_method=self.hparams["penalty_method"]
        )
        list_function_to_dict = [
            (
                self.call_alignment_method, DICT_STRATEGY_TO_METHOD,
                pd_reg * self.hparams["pd_lambda"]
            ),
            (
                self.call_alignment_method_full, DICT_STRATEGY_TO_METHOD_FULL,
                pd_reg * self.hparams["pd_lambda"]
            ),
            (self.call_alignment_method_fisher, FisherMMD.DICT_STRATEGY_TO_METHOD_FISHER, 1.),
            (
                self.call_alignment_method_fisher, FisherMMD.DICT_STRATEGY_TO_METHOD_FISHER_SQUARE,
                pd_reg * self.hparams["msd_lambda"]
            ),
            (
                self._get_mmdfeatures_loss, DICT_STRATEGY_TO_METHOD_FEATURES,
                self.hparams["mmd_lambda"]
            ),
        ]

        dict_residual = {}
        for function, dictstrategy, coeff in list_function_to_dict:
            if dictstrategy.get(strategy) is not None:
                _dict_output, _loss_align, _dict_residual = function(
                    all_z,
                    all_logits,
                    minibatches,
                    dict_residual=dict_residual,
                    method=dictstrategy[strategy],
                )
                dict_residual.update(_dict_residual)
                dict_output.update(_dict_output)

                loss_align += coeff * _loss_align

        assert list(dict_output.keys())

        return dict_output, loss_align

    def call_alignment_method_logits(self, all_z, all_logits, minibatches, method):
        all_y = torch.cat([b["y"] for b in minibatches])
        all_logits_per_domain = self._call_classifier_per_domain(
            all_z, minibatches, permute=(method < 0)
        )
        all_softmax = F.softmax(all_logits, dim=1)
        all_softmax_per_domain = F.softmax(all_logits_per_domain, dim=1)
        if method in [0]:
            loss_align = (
                all_softmax_per_domain * (all_softmax_per_domain /
                                          (all_softmax.detach() + 1e-8)).log()
            ).sum(dim=1).mean()
        elif method == 2:
            loss_align = -(all_softmax.detach() *
                           (all_softmax_per_domain + 1e-8).log()).sum(dim=1).mean()
        elif method == -1:
            loss_align = F.cross_entropy(all_logits_per_domain, all_y)
        elif method == 1:
            loss_align = -F.cross_entropy(all_logits_per_domain, all_y)
        return loss_align

    def call_alignment_method_fisher(
        self, all_z, all_logits, minibatches, method, dict_residual=None
    ):

        len_minibatches = [b["x"].shape[0] for b in minibatches]
        if "fmap_" + self.classifier.name in dict_residual:
            all_fmap = dict_residual["fmap_" + self.classifier.name]
        else:
            all_fmap = FisherMMD._get_fmap(
                self, all_logits, torch.cat([b["y"] for b in minibatches]), self.classifier
            )
            dict_residual["fmap_" + self.classifier.name] = all_fmap

        list_dict_grads_means, list_dict_grads_cov = FisherMMD.build_list_dict_grads(
            self, all_fmap, len_minibatches, all_y=None, method=method
        )

        dict_output, penalty = FisherMMD.compute_penalty_fisher(
            self, list_dict_grads_means, list_dict_grads_cov
        )

        if "square" in method:
            dict_output = {key + "_square": value for key, value in dict_output.items()}
        return dict_output, penalty, dict_residual

    def call_alignment_method_full(self, all_z, all_logits, minibatches, method, **kwargs):

        # values setter

        sum_dist_loss = 0
        dict_output = {}
        list_ntk = []
        dict_residual = {}
        list_classifiers = [self.classifier_per_domain[i] for i in range(self.num_domains)]
        if method != "ntk_domain":
            list_classifiers += [self.classifier]

        for i, classifier in enumerate(list_classifiers):
            if classifier != self.classifier:
                _all_logits = classifier(all_z)
            else:
                _all_logits = all_logits
            all_fmap = FisherMMD._get_fmap(
                self, _all_logits, torch.cat([b["y"] for b in minibatches]), classifier
            )
            dict_residual["fmap_" + classifier.name] = all_fmap
            ntk = self._get_ntk(all_fmap)
            if method == "ntk_detach" and classifier == self.classifier:
                assert classifier == self.classifier
                ntk = ntk.detach()
            list_ntk.append(ntk)

        ## distance as 1 - similarity
        sum_dist_loss = 0.
        count = 0
        for (i, j) in itertools.combinations(range(len(list_classifiers)), 2):
            ntk1, ntk2 = list_ntk[i], list_ntk[j]
            ntk_dist = 1 - kernel_similarity.center_cos(ntk1, ntk2)
            sum_dist_loss += ntk_dist
            count += 1
            if j == self.num_domains:
                key = "l_" + str(i) + "_ntk"
            else:
                key = "l_" + str(i) + "_" + str(j) + "_ntk"
            dict_output[key] = ntk_dist
        return dict_output, sum_dist_loss / count, dict_residual

    def call_alignment_method(self, all_z, all_logits, minibatches, method, **kwargs):
        # values setter
        all_fmap = FisherMMD._get_fmap(
            self, all_logits, torch.cat([b["y"] for b in minibatches]), self.classifier
        )

        sum_dist_loss = 0
        all_idx = 0
        dict_output = {}
        losses = torch.zeros(len(minibatches))
        losses_per_domain = torch.zeros(len(minibatches))
        for i, b in enumerate(minibatches):
            x = b["x"]
            bsize = x.shape[0]
            z = all_z[all_idx:all_idx + bsize]

            # ntk
            ## main classifier
            ntk = self._get_ntk(all_fmap[all_idx:all_idx + bsize])
            if method == "ntk_detach":
                ntk = ntk.detach()

            ## domain classifier
            logits_per_domain = self.classifier_per_domain[i](z)
            fmap_per_domain = FisherMMD._get_fmap(
                self, logits_per_domain, b["y"], self.classifier_per_domain[i]
            )

            ntk_per_domain = self._get_ntk(fmap=fmap_per_domain)

            ## distance as 1 - similarity
            ntk_dist = 1 - kernel_similarity.center_cos(ntk, ntk_per_domain)

            # l2diff
            logits = all_logits[all_idx:all_idx + bsize]
            if method.startswith("l2_detach"):
                logits = logits.detach()
            loss = F.cross_entropy(logits, b["y"], reduction="none")
            losses[i] = loss.mean()
            loss_per_domain = F.cross_entropy(logits_per_domain, b["y"], reduction="none")
            losses_per_domain[i] = loss_per_domain.mean()
            l2_diff = ((loss - loss_per_domain)**2).mean()
            l2_diff_filt = (
                (loss - loss_per_domain)**2 * (torch.where(loss > loss_per_domain, 1., 0.))
            ).mean()

            dict_output.update(
                {
                    "l_" + str(i): loss.mean(),
                    "l_" + str(i) + "_pd": loss_per_domain.mean(),
                    "l_" + str(i) + "_diff": l2_diff,
                    "l_" + str(i) + "_ntk": ntk_dist,
                    "l_" + str(i) + "_diff_filt": l2_diff_filt,
                }
            )
            all_idx += bsize
            if method.split("_")[0] == "ntk":
                sum_dist_loss += ntk_dist
            elif method.split("_")[0] == "l2":
                if method.split("_")[-1] == "filtered":
                    sum_dist_loss += l2_diff_filt
                else:
                    sum_dist_loss += l2_diff

        dict_output.update(
            {
                "l_var": ((losses - losses.mean())**2).mean(),
                "l_var_pd": ((losses_per_domain - losses_per_domain.mean())**2).mean()
            }
        )
        return dict_output, sum_dist_loss / len(minibatches), {}

    def _get_ntk(self, fmap):
        fmap = torch.cat(list(fmap.values()), dim=1)  # bsize, dims
        ntk = torch.einsum("an,bn->ab", fmap, fmap)
        return ntk

    def predict(self, batch):
        features = self.featurizer(batch["x"].cuda())
        predict_dict = {"main": self.classifier(features)}
        if self.hparams["lr_d"] != 0:
            for domain, classifier_per_domain in enumerate(self.classifier_per_domain):
                predict_dict.update({"net" + str(domain): classifier_per_domain(features)})
        return predict_dict


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean)**2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VRExema(VREx):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VRExema, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.ema_per_domain = [
            MovingAverageClean(self.hparams["ema"])
            for _ in range(num_domains)
        ]

    def update(self, minibatches, unlabeled=None):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        for domain_id in range(len(minibatches)):
            losses[domain_id] = self.ema_per_domain[domain_id].update({
                "loss": losses[domain_id]
            })["loss"]

        mean = losses.mean()
        penalty = ((losses - mean)**2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}




class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}

    # This commented "update" method back-propagates through the gradients of
    # the inner update, as suggested in the original MAML paper.  However, this
    # is twice as expensive as the uncommented "update" method, which does not
    # compute second-order derivatives, implementing the First-Order MAML
    # method (FOMAML) described in the original MAML paper.

    # def update(self, minibatches, unlabeled=None):
    #     objective = 0
    #     beta = self.hparams["beta"]
    #     inner_iterations = self.hparams["inner_iterations"]

    #     self.optimizer.zero_grad()

    #     with higher.innerloop_ctx(self.network, self.optimizer,
    #         copy_initial_weights=False) as (inner_network, inner_optimizer):

    #         for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
    #             for inner_iteration in range(inner_iterations):
    #                 li = F.cross_entropy(inner_network(xi), yi)
    #                 inner_optimizer.step(li)
    #
    #             objective += F.cross_entropy(self.network(xi), yi)
    #             objective += beta * F.cross_entropy(inner_network(xj), yj)

    #         objective /= len(minibatches)
    #         objective.backward()
    #
    #     self.optimizer.step()
    #
    #     return objective


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(self.num_domains):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, self.num_domains):
                penalty += torchutils.mmd(features[i], features[j], self.kernel_type)

        objective /= self.num_domains
        if self.num_domains > 1:
            penalty /= self.num_domains * (self.num_domains - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_lambda"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, minibatches, unlabeled=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = (self.ema * return_embedding + (1 - self.ema) * self.embeddings[env])

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs, num_classes, self.hparams["nonlinear_classifier"]
        )

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"], weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.sd_reg = hparams["sd_reg"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)

        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p**2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "penalty": penalty.item()}


class ANDMask(ERM):
    """
    Learning Explanations that are Hard to Vary [https://arxiv.org/abs/2009.00329]
    AND-Mask implementation from [https://github.com/gibipara92/learning-explanations-hard-to-vary]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]

    def update(self, minibatches, unlabeled=None):

        total_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)

        self.optimizer.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        self.optimizer.step()

        return {"loss": mean_loss.item()}

    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = mask.sum() / mask.numel()
            param.grad = mask * avg_grad
            param.grad *= 1.0 / (1e-10 + mask_t)

        return 0


class SANDMask(ERM):
    """
    TODO add link arxiv
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.tau = hparams["tau"]
        self.k = hparams["k"]
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'],
            betas=betas
        )

        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None):

        total_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            # TODO
            env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)

        self.optimizer.zero_grad()
        # TODO
        self.mask_grads(param_gradients, self.network.parameters())
        self.optimizer.step()
        self.update_count += 1

        return {'loss': mean_loss.item()}

    def mask_grads(self, gradients, params):
        '''
        TODO
        '''
        device = gradients[0][0].device
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            avg_grad = torch.mean(grads, dim=0)
            grad_signs = torch.sign(grads)
            gamma = torch.tensor(1.0).to(device)
            grads_var = grads.var(dim=0)
            grads_var[torch.isnan(grads_var)] = 1e-17
            lam = (gamma * grads_var).pow(-1)
            mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
            mask = torch.max(mask, torch.zeros_like(mask))
            mask[torch.isnan(mask)] = 1e-17
            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

class IGA(ERM):
    """
    Inter-environmental Gradient Alignment
    From https://arxiv.org/abs/2008.01883v2
    """

    def __init__(self, in_features, num_classes, num_domains, hparams):
        super(IGA, self).__init__(in_features, num_classes, num_domains, hparams)

    def update(self, minibatches, unlabeled=False):
        total_loss = 0
        grads = []
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grad = autograd.grad(
                env_loss, self.network.parameters(), retain_graph=True, create_graph=True)

            grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)
        mean_grad = autograd.grad(
            mean_loss, self.network.parameters(), retain_graph=True)

        # compute trace penalty
        penalty_value = 0
        for grad in grads:
            for g, mean_g in zip(grad, mean_grad):
                penalty_value += (g - mean_g).pow(2).sum()

        objective = mean_loss + self.hparams['penalty'] * penalty_value

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': mean_loss.item(), 'penalty': penalty_value.item()}

class SelfReg(ERM):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SelfReg, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        input_feat_size = self.featurizer.n_outputs
        hidden_size = (input_feat_size if input_feat_size == 2048 else input_feat_size * 2)

        self.cdpl = nn.Sequential(
            nn.Linear(input_feat_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_feat_size),
            nn.BatchNorm1d(input_feat_size),
        )

    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        lam = np.random.beta(0.5, 0.5)

        batch_size = all_y.size()[0]

        # cluster and order features into same-class group
        with torch.no_grad():
            sorted_y, indices = torch.sort(all_y)
            sorted_x = torch.zeros_like(all_x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = all_x[order]
            intervals = []
            ex = 0
            for idx, val in enumerate(sorted_y):
                if ex == val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            all_x = sorted_x
            all_y = sorted_y

        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)

        output = self.classifier(feat)

        # shuffle
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0
        for end in intervals:
            shuffle_indices = torch.randperm(end - ex) + ex
            shuffle_indices2 = torch.randperm(end - ex) + ex
            for idx in range(end - ex):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end

        # mixup
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3

        # regularization
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(feat, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(feat, feat_3)

        cl_loss = F.cross_entropy(output, all_y)
        C_scale = min(cl_loss.item(), 1.)
        loss = cl_loss + C_scale * (
            lam * (L_ind_logit + L_ind_feat) + (1 - lam) * (L_hdl_logit + L_hdl_feat)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class FisherMMD(Algorithm):
    CUSTOM_FORWARD = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(FisherMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
                name="main"
            )
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.register_buffer("update_count", torch.tensor([0]))

        if self.hparams["ema"] != 0:
            # parameters_names = [n for n, _ in self.classifier.named_parameters()]
            self.ma_grads_per_domain = [
                MovingAverage(alpha=self.hparams["ema"]) for _ in range(self.num_domains)
            ]
            self.ma_fisher_per_domain = [
                MovingAverage(alpha=self.hparams["ema"]) for _ in range(self.num_domains)
            ]
            self.ma_fishercov_per_domain = [
                MovingAverage(alpha=self.hparams["ema"]) for _ in range(self.num_domains)
            ]
        IRMAdv._init_optimizer(self)
        self.loss_backpack = extend(nn.CrossEntropyLoss(reduction='none'))

    def zero_grad(self):
        self.optimizer.zero_grad()

    DICT_STRATEGY_TO_METHOD_FISHER = {
        # 84: "fisher_mmd_centeredema",
        # 90: "fisher_mmd_emagrad",
        # 91: "fisher_mmd",
        # 91110: "fisher_mmd",
        # 101: "fisher_mmd_conditional",
        92: "fisher_mmd_centered",
        # 192: "fisher_mmd_centered",
        920: "fisher_mmd_centered_emagrad",
        # 92110: "fisher_mmd_centered",
        # deprecated
        # 82: "fisher_mmd_centered",
        # 87: "fisher_mmd_centered",
        # 88: "fisher_mmd_centered",
        # 89: "fisher_mmd",
        # 8286: "fisher_mmd_centered",
        # 80: "fisher_mmd",
        # 7180: "fisher_mmd",
        # 8081: "fisher_mmd"
    }

    DICT_STRATEGY_TO_METHOD_FISHER_SQUARE = {
        # 86: "fisher_square_centered",
        91110: "fisher_square",
        92110: "fisher_square_centered",
        # 8081: "fisher_square",
        # 8286: "fisher_square_centered"
    }

    def _call_penalty(self, all_fmap, len_minibatches, all_y, method):

        list_dict_grads_means, list_dict_grads_cov = self.build_list_dict_grads(
            all_fmap, len_minibatches, all_y=all_y, method=method
        )

        if self.hparams["strategy"] < 80:
            raise NotImplementedError
        else:
            # if self.hparams["strategy"] in [88, 89]:
            #     assert self.hparams["mean_lambda"] == 0
            if "emagrad" in method:
                assert self.hparams["mean_lambda"] != 0
            # if self.hparams["strategy"] == 192:
            #     dict_output, penalty = self.compute_penalty_fisher_1v1(
            #         list_dict_grads_means,
            #         list_dict_grads_cov,
            #     )
            # else:
            dict_output, penalty = self.compute_penalty_fisher(
                list_dict_grads_means,
                list_dict_grads_cov,
            )
        return dict_output, penalty

    def update(self, minibatches, unlabeled=False):
        len_minibatches = [b["x"].shape[0] for b in minibatches]
        all_x = torch.cat([b["x"] for b in minibatches])
        all_y = torch.cat([b["y"] for b in minibatches])

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)
        dict_output = {}
        all_fmap = self._get_fmap(all_logits, all_y, self.classifier)

        penalty = 0
        if self.hparams["strategy"] in self.DICT_STRATEGY_TO_METHOD_FISHER:
            _dict_output, _penalty = self._call_penalty(
                all_fmap, len_minibatches, all_y, method=self.DICT_STRATEGY_TO_METHOD_FISHER[self.hparams["strategy"]])

            penalty += _penalty
            dict_output.update(_dict_output)
        # if self.hparams["strategy"] in self.DICT_STRATEGY_TO_METHOD_FISHER_SQUARE:
        #     _dict_output, _penalty = self._call_penalty(
        #         all_fmap, len_minibatches, all_y, method=self.DICT_STRATEGY_TO_METHOD_FISHER_SQUARE[self.hparams["strategy"]])
        #     penalty += self.hparams["msd_lambda"] * _penalty
        #     dict_output.update({"sq_" + key: value for key, value in _dict_output.items()})

        all_loss = F.cross_entropy(all_logits, all_y)

        coeff_reg = self.get_current_consistency_weight()
        if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            print("Finally reach penalty_anneal_iters: " + str(self.update_count))
            if int(self.hparams["penalty_method"]) in [2, 4]:
                IRMAdv._init_optimizer(self)
            else:
                raise ValueError("penalty_method")

        objective = all_loss + coeff_reg * penalty
        # if self.hparams["penalty_method"] in [3, 4]:
        #     raise ValueError("penalty_method")
        #     objective /= 1 + coeff_reg * (self.hparams["mmd_lambda"] + self.hparams["mean_lambda"])

        self.zero_grad()
        objective.backward()
        self.optimizer.step()

        dict_output.update({
            "l_gen": objective,
            'l': all_loss,
        })
        self.update_count += 1
        return {k: v.item() for k, v in dict_output.items()}

    def predict(self, batch):
        return {"main": self.network(batch["x"].cuda())}

    def get_current_consistency_weight(self):

        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            coeff_reg = 1
        else:
            coeff_reg = losses.get_current_consistency_weight(
                epoch=self.update_count.item(),
                consistency_rampup=self.hparams["penalty_anneal_iters"],
                epoch_start_consistency_rampup=0,
                penalty_anneal_method=self.hparams["penalty_method"]
            )
        return coeff_reg

    def get_tb_dict(self):
        dict_output = {
            "coeff_reg": self.get_current_consistency_weight(),
            "mmd_lambda": self.hparams["mmd_lambda"],
            "mean_lambda": self.hparams["mean_lambda"]
        }
        if self.hparams["strategy"] != "none":
            dict_output["strategy"] = self.hparams["strategy"]
        return dict_output

    def build_list_dict_grads(self, all_fmap, len_minibatches, all_y, method):
        list_dict_grads_means = [defaultdict(lambda: {}) for _ in range(self.num_domains)]
        list_dict_grads_cov = [defaultdict(lambda: {}) for _ in range(self.num_domains)]
        set_possible_y = set(list(all_y.cpu().detach().numpy().squeeze()))
        for n, fmap in all_fmap.items():
            all_idx = 0
            for i, bsize in enumerate(len_minibatches):
                env_grads = fmap[all_idx:all_idx + bsize]
                if "conditional" in method:
                    minibatch_y = all_y[all_idx:all_idx + bsize]
                    for possible_y in set_possible_y:
                        env_grads_cond = env_grads[minibatch_y == possible_y]
                        if env_grads_cond.size(0) == 0:
                            env_grads_cond = env_grads
                        list_dict_grads_means[i][possible_y][n], list_dict_grads_cov[i][possible_y][
                            n] = self.from_envgrads_to_dict(
                                env_grads=env_grads_cond, method=method
                            )
                else:
                    list_dict_grads_means[i]["non_conditional"][n], list_dict_grads_cov[i][
                        "non_conditional"][n] = self.from_envgrads_to_dict(env_grads, method)
                all_idx += bsize

        # moving average
        if self.hparams["ema"] != 0:
            for i, _ in enumerate(len_minibatches):
                if "square" in method:
                    list_dict_grads_cov[i] = self.ma_fishercov_per_domain[i].update(
                        list_dict_grads_cov[i]
                    )
                else:
                    if "emagrad" in method:
                        ema_dict_data = self.ma_grads_per_domain[i].update(list_dict_grads_means[i])
                        list_dict_grads_means[i] = ema_dict_data
                    list_dict_grads_cov[i] = self.ma_fisher_per_domain[i].update(
                        list_dict_grads_cov[i]
                    )

        return list_dict_grads_means, list_dict_grads_cov

    def from_envgrads_to_dict(self, env_grads, method):

        means = env_grads.mean(dim=0, keepdim=True)
        if "centered" in method:
            if "centeredema" in method:
                raise ValueError(method)
            # if ("centeredema" in method and self.ma_grads_per_domain[i].updates):
            #     env_grads = math.sqrt(self.hparams["ema"]
            #                          ) * (env_grads - self.ma_grads_per_domain[i].parameter[n])
            # else:
            env_grads = env_grads - means

        if "square" in method:
            covariance_env_grads = torch.einsum("na,nb->ab", env_grads, env_grads)
            covs = (covariance_env_grads - torch.diagonal(covariance_env_grads)) / env_grads.size(0) / (env_grads.size(1) - 1)
            # covs = covariance_env_grads.reshape(-1) / (env_grads.size(0))# * env_grads.size(1))
        else:
            covs = (env_grads).pow(2).mean(dim=0)
        return means, covs

    # def compute_penalty_fisher_1v1(self, list_dict_grads_means, list_dict_grads_cov):
    #     penalty_mean = 0
    #     penalty_cov = 0
    #     dict_output = {}

    #     for i in range(self.num_domains):
    #         for key in list_dict_grads_cov[0]:
    #             dict_output["l2_fisher_norm_" + str(i) + "_" + str(key)] = fisher_metrics.l2norm(
    #                 list_dict_grads_cov[i][key]
    #             )
    #             for j in range(i + 1, self.num_domains):
    #                 i_penalty_mean = fisher_metrics.distance_between_dicts(
    #                     dict1=list_dict_grads_means[i][key],
    #                     dict2=list_dict_grads_means[j][key],
    #                     strategy=self.hparams['strategy_mean']
    #                 )
    #                 penalty_mean += i_penalty_mean

    #                 # reg cov gradients
    #                 i_penalty_cov = fisher_metrics.distance_between_dicts(
    #                     dict1=list_dict_grads_cov[i][key],
    #                     dict2=list_dict_grads_cov[j][key],
    #                     strategy=self.hparams['strategy_cov']
    #                 )
    #                 penalty_cov += i_penalty_cov

    #     # len(list_dict_grads_cov[0])
    #     if self.hparams['strategy_cov'].endswith("norm"):
    #         num_keys = self.num_classes
    #     elif self.hparams['strategy_cov'].endswith("perm"):
    #         num_keys = (self.num_domains - 1) / 2
    #     else:
    #         num_keys = 1
    #     penalty_cov /= (self.num_domains * num_keys)
    #     penalty_mean /= (self.num_domains * num_keys)
    #     dict_output.update({"l_fisher_mean": penalty_mean, "l_fisher_cov": penalty_cov})
    #     penalty = (
    #         self.hparams["mmd_lambda"] * penalty_cov + self.hparams["mean_lambda"] * penalty_mean
    #     )
    #     return dict_output, penalty

    def compute_penalty_fisher(self, list_dict_grads_means, list_dict_grads_cov):
        penalty_mean = 0
        penalty_cov = 0
        dict_output = {}

        if self.hparams["mean_lambda"] == 0:
            dict_grads_means = None
        else:
            dict_grads_means = {
            key: {
                param: torch.stack(
                    [list_dict_grads_means[i][key][param] for i in range(self.num_domains)], dim=0
                    ).mean(dim=0) for param in list_dict_grads_means[0][key]
                } for key in list_dict_grads_means[0]
            }
        dict_grads_cov = {
            key: {
                param: torch.stack(
                    [list_dict_grads_cov[i][key][param] for i in range(self.num_domains)], dim=0
                ).mean(dim=0) for param in list_dict_grads_cov[0][key]
            } for key in list_dict_grads_cov[0]
        }
        for i in range(self.num_domains):
            for key in list_dict_grads_cov[0]:
                # dict_output["l2_fisher_norm_" + str(i) + "_" + str(key)] = fisher_metrics.l2norm(
                #     list_dict_grads_cov[i][key]
                # )
                if self.hparams["mean_lambda"] != 0:
                    i_penalty_mean = fisher_metrics.distance_between_dicts(
                        dict1=list_dict_grads_means[i][key],
                        dict2=dict_grads_means[key],
                        strategy=self.hparams['strategy_mean']
                    )
                    penalty_mean += i_penalty_mean

                # reg cov gradients
                i_penalty_cov = fisher_metrics.distance_between_dicts(
                    dict1=list_dict_grads_cov[i][key],
                    dict2=dict_grads_cov[key],
                    strategy=self.hparams['strategy_cov']
                )
                penalty_cov += i_penalty_cov

        # len(list_dict_grads_cov[0])
        if self.hparams['strategy_cov'].endswith("norm"):
            num_keys = self.num_classes
        elif self.hparams['strategy_cov'].endswith("perm"):
            num_keys = (self.num_domains - 1) / 2
        else:
            num_keys = 1
        penalty_cov /= (self.num_domains * num_keys)
        penalty_mean /= (self.num_domains * num_keys)
        dict_output.update({
            "l_fisher_cov": penalty_cov,
            })
        penalty = (
            self.hparams["mmd_lambda"] * penalty_cov
        )
        if self.hparams["mean_lambda"] != 0:
            dict_output["l_fisher_mean"] = penalty_mean
            penalty += self.hparams["mean_lambda"] * penalty_mean
        return dict_output, penalty

    def _get_fmap(self, logits, y, classifier):
        self.zero_grad()
        if self.hparams["grad_wrt"] == "loss":
            tensortoderive = self.loss_backpack(logits, y).sum()
        # elif self.hparams["grad_wrt"] == "lossmax":
        #     # fails as it force more confidence on easy labels<
        #     raise ValueError
        #     _, y = logits.max(dim=1)
        #     tensortoderive = self.loss_backpack(logits, y.detach()).sum()
        # elif self.hparams["grad_wrt"] == "maxlogit":
        #     tensortoderive, _ = logits.max(dim=1)
        #     tensortoderive = tensortoderive.mean()
        # elif self.hparams["grad_wrt"] == "truelogit":
        #     tensortoderive = logits.gather(1, y.view(-1, 1)).view(-1).mean()
        else:
            raise ValueError(self.hparams["grad_wrt"])

        with backpack(BatchGrad()):
            tensortoderive.backward(
                inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
            )

        fmap = OrderedDict(
            {
                n: p.grad_batch.clone().view(p.grad_batch.size(0), -1)
                for n, p in classifier.named_parameters()
            }
        )  # num_params => bsize
        return fmap
    def predict(self, batch):
        return {"main": self.network(batch["x"].cuda())}




class LFF(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    CUSTOM_FORWARD = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.featurizer_weak = networks.Featurizer(input_shape, self.hparams)
        self.classifier_weak = networks.Classifier(
            self.featurizer_weak.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )

        self.network_weak = nn.Sequential(self.featurizer_weak, self.classifier_weak)

        self.network_strong = nn.Sequential(
            networks.Featurizer(input_shape, self.hparams),
            networks.Classifier(
                self.featurizer_weak.n_outputs,
                num_classes,
                self.hparams["nonlinear_classifier"],
            ),
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        if self.hparams["weak_gce"]:
            self.weak_loss = GeneralizedCELoss(q=hparams["weight_q"])
        else:
            self.weak_loss = nn.CrossEntropyLoss()

        self.regular_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        self.sample_loss_ema_b = EMA(alpha=0.7)
        self.sample_loss_ema_d = EMA(alpha=0.7)
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([b["x"] for b in minibatches])
        all_y = torch.cat([b["y"] for b in minibatches])
        all_index = torch.cat([b["index"] for b in minibatches]).cpu().numpy()
        logits_b = self.network_weak(all_x)
        logits_d = self.network_strong(all_x)

        if self.hparams["reweighting"]:
            loss_b_weight = (F.cross_entropy(logits_b, all_y, reduction="none").cpu().detach())
            loss_d_weight = (F.cross_entropy(logits_d, all_y, reduction="none").cpu().detach())
            self.sample_loss_ema_b.update(loss_b_weight, all_index, all_y)
            self.sample_loss_ema_d.update(loss_d_weight, all_index, all_y)
            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[all_index].copy()
            loss_d = self.sample_loss_ema_d.parameter[all_index].copy()

            for c in range(self.num_classes):
                class_index = np.where(all_y.cpu().numpy() == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c)
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            if np.isnan(loss_weight.mean().item()):
                raise NameError("loss_weight")
        else:
            loss_weight = 1.0

        loss_b_update = self.weak_loss(logits_b, all_y)
        loss_d_update = (
            F.cross_entropy(logits_d, all_y, reduction="none") * torch.tensor(loss_weight).cuda()
        )

        loss = loss_b_update.mean() + loss_d_update.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "weak_loss": loss_b_update.mean().item(),
            "strong_loss": loss_d_update.mean().item(),
        }

    def predict(self, batch):
        predict_strong = self.network_strong(batch["x"].cuda())
        return {
            "weak": self.network_weak(batch["x"].cuda()),
            "strong": predict_strong,
            "main": predict_strong,
        }




class KernelDiversity(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    CUSTOM_FORWARD = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.hparams = hparams

        # WEAK
        self.featurizer_weak = networks.Featurizer(input_shape, self.hparams)
        self.classifier_weak = extend(
            networks.Classifier(
                self.featurizer_weak.n_outputs,
                num_classes,
                self.hparams["nonlinear_classifier"],
            )
        )
        # self.network_weak = nn.Sequential(self.featurizer_weak, self.classifier_weak)

        # STRONG
        self.featurizer_strong = networks.Featurizer(input_shape, self.hparams)
        self.classifier_strong = extend(
            networks.Classifier(
                self.featurizer_weak.n_outputs,
                num_classes,
                self.hparams["nonlinear_classifier"],
            )
        )
        # self.network_strong = nn.Sequential(
        #     self.featurizer_strong, self.classifier_strong
        # )

        # OPTIM, LOSS

        if self.hparams["weak_gce"]:
            self.weak_loss = GeneralizedCELoss(q=hparams["weight_q"])
        else:
            self.weak_loss = nn.CrossEntropyLoss()
        self.regular_loss = extend(nn.CrossEntropyLoss())

        if self.hparams["reweighting"]:
            self.sample_loss_ema_b = EMA(alpha=0.7)
            self.sample_loss_ema_d = EMA(alpha=0.7)

        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([b["x"] for b in minibatches])
        all_y = torch.cat([b["y"] for b in minibatches])
        # all_index = torch.cat([b["index"] for b in minibatches]).cpu().numpy()

        features_weak = self.featurizer_weak(all_x)
        features_strong = self.featurizer_strong(all_x)

        logits_b = self.classifier_weak(features_weak)
        logits_d = self.classifier_strong(features_strong)

        # # TODO: test if this hack is equivalent (stop grad before features)
        # logits_b_nograd = self.classifier_weak(features_weak.detach())
        # logits_d_nograd = self.classifier_weak(features_strong.detach())
        loss_b_nograd = self.regular_loss(logits_b, all_y)
        loss_d_nograd = self.regular_loss(logits_d, all_y)
        loss_nograd = loss_b_nograd + loss_d_nograd

        bsize = len(all_x)
        # TODO: put requires_grad on featurizer to false before doing this
        #    for faster backward, and then to true again ?

        # Compute Kernel
        if self.hparams["kernel"] == "ntk":
            with backpack(BatchGrad()):
                loss_nograd.backward(
                    retain_graph=True,
                    create_graph=True,
                    inputs=list(self.classifier_strong.parameters()) +
                    list(self.classifier_weak.parameters())
                )
            # get gradients
            if self.hparams["kernel_on"] == "classifier":
                fmap_weak = [
                    p.grad_batch.view(bsize, -1) for p in self.classifier_weak.parameters()
                ]  # bsize, dims*
                fmap_strong = [
                    p.grad_batch.view(bsize, -1) for p in self.classifier_strong.parameters()
                ]  # bsize, dims*
                fmap_weak = torch.cat(fmap_weak, dim=1)  # bsize, dims
                fmap_strong = torch.cat(fmap_strong, dim=1)  # bsize, dims
            elif self.hparams["kernel_on"] == "all":
                fmap_weak = [
                    p.grad_batch.view(bsize, -1) for p in self.parameters()
                ]  # bsize, dims*
                fmap_strong = [
                    p.grad_batch.view(bsize, -1) for p in self.parameters()
                ]  # bsize, dims*
                fmap_weak = torch.cat(fmap_weak, dim=1)  # bsize, dims
                fmap_strong = torch.cat(fmap_strong, dim=1)  # bsize, dims
        elif self.hparams["kernel"] == "features":
            fmap_weak = features_weak
            fmap_strong = features_strong

        if self.hparams["detach_weak"]:
            fmap_weak = fmap_weak.detach()

        # Compute Similarity
        if self.hparams["similarity"] == "dot":
            kernel_weak = torch.einsum("an,bn->ab", fmap_weak, fmap_weak)
            kernel_weak = kernel_weak - torch.diagonal(kernel_weak)
            kernel_strong = torch.einsum("an,bn->ab", fmap_strong, fmap_strong)
            kernel_strong = kernel_strong - torch.diagonal(kernel_strong)
            similarity = kernel_similarity.dot(kernel_strong, kernel_weak)
        elif self.hparams["similarity"] == "center-dot":
            kernel_weak = torch.einsum("an,bn->ab", fmap_weak, fmap_weak)
            kernel_strong = torch.einsum("an,bn->ab", fmap_strong, fmap_strong)
            # kernel_weak = kernel_weak - torch.mean(kernel_weak, 0)
            # kernel_strong = kernel_strong - torch.mean(kernel_strong, 0)
            # similarity = (kernel_weak * kernel_strong).sum()
            similarity = kernel_similarity.center_dot(kernel_weak, kernel_strong)
        elif self.hparams["similarity"] == "cos":
            kernel_weak = torch.einsum("an,bn->ab", fmap_weak, fmap_weak)
            kernel_weak = kernel_weak - torch.diagonal(kernel_weak)
            kernel_strong = torch.einsum("an,bn->ab", fmap_strong, fmap_strong)
            kernel_strong = kernel_strong - torch.diagonal(kernel_strong)
            # similarity = (kernel_weak * kernel_strong).sum()
            # similarity = (similarity / kernel_weak.norm() / kernel_strong.norm())
            similarity = kernel_similarity.cos(kernel_weak, kernel_strong)
        elif self.hparams["similarity"] == "center-cos":
            kernel_weak = torch.einsum("an,bn->ab", fmap_weak, fmap_weak)
            kernel_strong = torch.einsum("an,bn->ab", fmap_strong, fmap_strong)
            similarity = kernel_similarity.center_cos(kernel_weak, kernel_strong)
            # kernel_weak = kernel_weak - torch.mean(kernel_weak, 0)
            # kernel_strong = kernel_strong - torch.mean(kernel_strong, 0)
            # similarity = (kernel_weak * kernel_strong).sum()
            # similarity = (similarity / kernel_weak.norm() / kernel_strong.norm())
        elif self.hparams["similarity"] == "linear-cka":
            similarity = torch.norm(torch.mm(fmap_weak.t(), fmap_strong))
            similarity = similarity / torch.norm(torch.mm(fmap_weak.t(), fmap_weak))
            similarity = similarity / torch.norm(torch.mm(fmap_strong.t(), fmap_strong))
        elif self.hparams["similarity"] == "linear-cka-transpose":
            similarity = torch.norm(torch.mm(fmap_weak, fmap_strong.t()))
            similarity = similarity / torch.norm(torch.mm(fmap_weak, fmap_weak.t()))
            similarity = similarity / torch.norm(torch.mm(fmap_strong, fmap_strong.t()))

        if self.hparams["similarity_result"] == "square":
            similarity = similarity**2
        elif self.hparams["similarity_result"] == "none":
            pass
        elif self.hparams["similarity_result"] == "abs":
            similarity = similarity.abs()
        elif self.hparams["similarity_result"] == "relu":
            similarity = F.relu(similarity)
        else:
            raise ValueError(self.hparams["similarity_result"])

        loss_similarity = similarity

        self.optimizer.zero_grad()
        # for p in self.featurizer_strong.parameters():
        #     p.requires_grad = True
        # for p in self.featurizer_weak.parameters():
        #     p.requires_grad = True

        # logits_b = self.classifier_weak(features_weak)
        # logits_d = self.classifier_strong(features_strong)

        # regular loss
        loss_b_update = self.weak_loss(logits_b, all_y)

        # reweighting
        if self.hparams["reweighting"]:
            all_index = torch.cat([b["index"] for b in minibatches]).cpu().numpy()
            loss_b_weight = F.cross_entropy(logits_b, all_y, reduction="none").cpu().detach()
            loss_d_weight = F.cross_entropy(logits_d, all_y, reduction="none").cpu().detach()
            self.sample_loss_ema_b.update(loss_b_weight, all_index, all_y)
            self.sample_loss_ema_d.update(loss_d_weight, all_index, all_y)
            loss_b = self.sample_loss_ema_b.parameter[all_index].copy()
            loss_d = self.sample_loss_ema_d.parameter[all_index].copy()
            for c in range(self.num_classes):
                class_index = np.where(all_y.cpu().numpy() == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c)
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            loss_d_update = (
                F.cross_entropy(logits_d, all_y, reduction="none") *
                torch.tensor(loss_weight).cuda()
            )
        else:
            loss_d_update = F.cross_entropy(logits_d, all_y, reduction="none")

        loss = (
            loss_b_update.mean() + loss_d_update.mean() +
            self.hparams["similarity_weight"] * loss_similarity
        )

        # breakpoint()
        loss.backward()
        # print(self.featurizer_strong.features[0].weight.grad)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "weak_loss": loss_b_update.mean().item(),
            "strong_loss": loss_d_update.mean().item(),
            "sim_loss": loss_similarity.item(),
        }

    def predict(self, batch):
        logit_weak = self.classifier_weak(self.featurizer_weak(batch["x"].cuda()))
        logit_strong = self.classifier_strong(self.featurizer_strong(batch["x"].cuda()))
        return {
            "weak": logit_weak,
            "strong": logit_strong,
            "ensemble": logit_weak + logit_strong,
            "main": logit_strong,
        }





class EnsembleKernelDiversity(Algorithm):
    CUSTOM_FORWARD = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.num_classifiers = self.hparams["num_classifiers"]
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        print(self.featurizer)

        self.classifiers = nn.ModuleList([
                extend(networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams["nonlinear_classifier"],
                    hparams=self.hparams,
                    ))
                 for _ in range((self.num_classifiers))
        ])

        print(self.classifiers)

        if self.hparams["ntk_loss"] == "cross-entropy":
            self.backpack_loss = extend(nn.CrossEntropyLoss(reduction="sum"))
        elif self.hparams["ntk_loss"] == "bce":
            self.backpack_loss = extend(nn.BCEWithLogitsLoss(reduction="sum"))


        if self.hparams["loss"] == "cross-entropy":
            self.loss = nn.CrossEntropyLoss(reduction="sum")
        elif self.hparams["loss"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="sum")

        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.spectral_decoupling = self.hparams["spectral_decoupling"]

        self.true_logits_classifier = nn.Sequential(
            nn.Linear(self.featurizer.n_outputs, self.featurizer.n_outputs),
            nn.ReLU(),
            nn.Linear(self.featurizer.n_outputs, num_classes)
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([b["x"] for b in minibatches])
        all_y = torch.cat([b["y"] for b in minibatches])

        if "biases" in minibatches[0]:
            biases = {key: torch.cat([b["biases"][key] for b in minibatches]) for key in minibatches[0]["biases"]}
        else:
            biases = dict()
        # all_colors = torch.cat([b["colors"] for b in minibatches])
        # all_shapes = torch.cat([b["shapes"] for b in minibatches])

        gt_kernel = (all_y[None, :] == all_y[:, None]).float()
        gt_kernel_biases = {key: (biases[key][None, :] == biases[key][:, None]).float() for key in biases}
        distances_with_gt = None

        steps = [b["step"] for b in minibatches]
        steps = steps[0]

        if self.hparams["freeze_featurizer"]:
            with torch.no_grad():
                self.featurizer.eval()
                features = self.featurizer(all_x).detach()
        else:
            features = self.featurizer(all_x)

        logits = []

        for classif in self.classifiers:
            logits.append(classif(features))

        # Compute Kernel
        if self.hparams["similarity_weight"] == 0.0:
            loss_similarity = torch.tensor(0.0)
        else:
            # if self.hparams["kernel"] == "ntk" or self.hparams["kernel"] ==  "weights":

            if self.hparams["kernel"] == "ntk":
                if self.hparams["kernel_on"] == "classifier":
                    if self.hparams["ntk_loss"] == "max-sum":
                        losses = [logit.max(1).values.sum() for logit in logits]
                    elif self.hparams["ntk_loss"] == "correct-sum":
                        losses = [torch.gather(logit, 0, all_y[:, None]).sum() for logit in logits]
                    else:
                        losses = [self.backpack_loss(logit, all_y) for logit in logits]
                    sum_loss = torch.stack(losses).mean()
                    bsize = len(all_x)
                    with backpack(BatchGrad()):
                        sum_loss.backward(retain_graph=True, create_graph=True, inputs=list(self.classifiers.parameters()))
                    # get gradients
                    fmaps = [
                        torch.cat([p.grad_batch.view(bsize, -1) for p in classifier.parameters()], dim=1)
                        for classifier in self.classifiers
                    ]   # (bsize, num_params)

                    if self.hparams["no_diversity_first_model"]:
                        fmaps[0] = fmaps[0].detach()

                elif self.hparams["kernel_on"] == "logits":
                    fmaps = logits

                # center fmaps ?
                if self.hparams["center_gradients"] == "all":
                    fmaps = [fmap - fmap.mean(dim=1, keepdim=True) for fmap in fmaps]
                elif self.hparams["center_gradients"] == "classes":
                    fmaps = torch.stack(fmaps)  # num_classif, bsize, dim (num_weights)
                    for val in torch.unique(all_y):
                        indexes = (all_y==val).nonzero().squeeze()
                        mean = fmaps[:, indexes].mean(dim=2, keepdim=True)
                        fmaps[:, indexes] = fmaps[:, indexes]  - mean

                if self.hparams["normalize_gradients"] == True:
                    fmaps = [fmap / fmap.norm(dim=1, keepdim=True) for fmap in fmaps]

                # Compute Similarity
                kernels = [fmap @ fmap.t() for fmap in fmaps]
                if self.hparams["difference_gt_kernel"] == "l2":
                    kernels = [(ker.abs() - gt_kernel)**2 for ker in kernels]
                if self.hparams["difference_gt_kernel"] == "l1":
                    kernels = [torch.abs(ker.abs() - gt_kernel) for ker in kernels]

                distances_with_gt = [kernel_similarity.cos(k, gt_kernel) for k in kernels]
                distances_with_gt_biases = {key: [kernel_similarity.cos(k, gt_kernel_biases[key]) for k in kernels] for key in biases}
                # distances_with_gt_color = [kernel_similarity.cos(k, gt_kernel_color) for k in kernels]
                # distances_with_gt_shape = [kernel_similarity.cos(k, gt_kernel_shape) for k in kernels]

                if self.hparams["similarity"] == "dot":
                    ks = torch.stack(kernels, dim=0)
                    similarity = kernel_similarity.batch_dot(ks)
                if self.hparams["similarity"] == "dot-diagonal":
                    kernels = torch.stack([k - torch.diagonal(k) for k in kernels], dim=0)
                    similarity = kernel_similarity.batch_dot(kernels)
                elif self.hparams["similarity"] == "center-dot":
                    ks = torch.stack(kernels, dim=0)
                    similarity = kernel_similarity.batch_center_dot(ks)
                elif self.hparams["similarity"] == "cos":
                    ks = torch.stack(kernels, dim=0)
                    similarity = kernel_similarity.batch_cos(ks)
                elif self.hparams["similarity"] == "center-cos":
                    ks = torch.stack(kernels, dim=0)
                    similarity = kernel_similarity.batch_center_cos(ks)
                elif self.hparams["similarity"] == "l2":
                    ks = torch.stack(kernels, dim=0)
                    similarity = kernel_similarity.batch_l2(ks)
                elif self.hparams["similarity"] == "l1":
                    ks = torch.stack(kernels, dim=0)
                    similarity = kernel_similarity.batch_l1(ks)

                if self.hparams["similarity_result"] == "square":
                    similarity = similarity**2
                elif self.hparams["similarity_result"] == "none":
                    pass
                elif self.hparams["similarity_result"] == "abs":
                    similarity = similarity.abs()
                elif self.hparams["similarity_result"] == "relu":
                    similarity = F.relu(similarity)
                else:
                    raise ValueError(self.hparams["similarity_result"])
                loss_similarity = similarity

            elif self.hparams["kernel"] == "weights":
                weights =  torch.stack([classif.weight for classif in self.classifiers], dim=0)  # num_classif, num_out, num_in
                # normalize directions in num_out
                weights = weights / weights.norm(dim=2, keepdim=True)

                similarity = torch.einsum("noi,moi->nm", [weights, weights])
                # breakpoint()
                loss_similarity = similarity.sum()

            elif self.hparams["kernel"] == "grads":
                weights =  [classif.weight for classif in self.classifiers]  # num_classif, num_out, num_in
                losses = [self.backpack_loss(logit, all_y) for logit in logits]
                grads = torch.autograd.grad(losses, weights, create_graph=True)
                grads = torch.stack(grads, dim=0)  # num_classif, num_out, num_in
                # normalize directions in num_out
                grads = grads / grads.norm(dim=2, keepdim=True)
                similarity = torch.einsum("noi,moi->nm", [grads, grads])
                # breakpoint()
                loss_similarity = similarity.sum()


            elif self.hparams["kernel"] == "teney":
                fmaps = []
                for logit in logits:
                    pred, _ = torch.max(logit, dim=1)  # (bsize,)
                    # logit: (bsize, num_classes)
                    g = torch.autograd.grad(
                        pred.sum(0), features, retain_graph=True, create_graph=True
                    )   # (bsize, dim_feats)
                    assert len(g) == 1
                    g = torch.abs(g[0])
                    # grads[:, :, i] = g[0]
                    fmaps.append(g)
                fmaps = torch.stack(fmaps, dim=0)  # (num_classif, bsize, dim)
                distances = torch.einsum("nbd,mbd->nm", fmaps, fmaps)  # (num_classif, num_classif)
                distances = distances - torch.diag(torch.diag(distances))
                loss_similarity = distances.sum() / 2

            # elif self.hparams["kernel"] == "weights":
            # weights = [c.weight for c in  self.classifiers]
            # distances = torch.einsum("nbd,mbd->nm", weights, weights)  # (num_classif, num_classif)


        if self.hparams["loss"] == "bce":
            all_y = one_hot_embedding(all_y, self.num_classes)

        losses = [self.loss(logit, all_y) for logit in logits]

        # Spectral decoupling
        if self.spectral_decoupling != 0.0:
            sd_loss = torch.mean(torch.stack([self.spectral_decoupling * logit.norm(dim=1).mean() for logit in logits]))
        else:
            sd_loss = torch.tensor(0.0)
        sum_loss = torch.stack(losses).mean()

        loss = sum_loss + sd_loss

        if steps < self.hparams["similarity_schedule_start_at"]:
            similarity_weight = 0.0
        else:
            if self.hparams["similarity_schedule"] == "none":
                similarity_weight = self.hparams["similarity_weight"]
            elif self.hparams["similarity_schedule"] == "linear":
                similarity_weight = self.hparams["similarity_weight"] + (steps-self.hparams["similarity_schedule_start_at"]) * self.hparams["similarity_schedule_param1"]
            elif self.hparams["similarity_schedule"] == "exp":
                similarity_weight = self.hparams["similarity_weight"] * (self.hparams["similarity_schedule_param1"] ** (steps-self.hparams["similarity_schedule_start_at"]))
            else:
                raise ValueError(self.hparams["similarity_schedule"])

        if similarity_weight != 0.0:
            loss = loss + similarity_weight * loss_similarity

        # TRUE LABELS CLASSIFIER
        if "shape" in biases:
            logits_true_labels = self.true_logits_classifier(features.detach())
            loss_true_labels = F.cross_entropy(logits_true_labels, biases["shape"].long())
            loss = loss + loss_true_labels
        else:
            loss_true_labels = torch.tensor(0.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        out = {
            "loss": loss.item(),
            "regular_loss": sum_loss.item(),
            "sd_loss": sd_loss.item(),
            "sim_loss": loss_similarity.item(),
            "loss_true_labels": loss_true_labels.item()
        }
        if distances_with_gt is not None:
            for i in range(len(distances_with_gt)):
                out[f"sim_with_gt_classif-{i}"] = distances_with_gt[i].item()
                for key in biases:
                    out[f"sim_with_gt_{key}_classif-{i}"] = distances_with_gt_biases[key][i].item()
        return out

    def predict(self, batch):
        feats = self.featurizer(batch["x"].cuda())
        logits = [classif(feats) for classif in self.classifiers]
        true_logits = self.true_logits_classifier(feats)
        ensemble_logits = sum(logits)
        results = {
            f"classif-{i}": logits[i] for i in range(len(logits))
        }
        results["ensemble"] = ensemble_logits
        results["pred_true_logits"] = true_logits
        return results

    @staticmethod
    def accuracy(network, loader, weights, device, num_classes=None, class_names=None):
        network.eval()
        corrects = defaultdict(int)  # key -> int
        corrects_biases = defaultdict(lambda: defaultdict(int))
        # corrects_color = defaultdict(int)
        # corrects_shape = defau
        totals = defaultdict(int)    # key -> int
        results = dict()
        preds =  dict()
        ys = []
        results_bias = dict()

        # if Precision is not None and num_classes is not None:
        #     per_class_precision = defaultdict(lambda: Precision(num_classes=num_classes, average="none").to(device))
        #     per_class_recall =  defaultdict(lambda: Recall(num_classes=num_classes, average="none").to(device))
        # else:
        per_class_precision, per_class_recall = None, None

        with torch.no_grad():
            for batch in loader:
                logits = network.predict(batch)
                batch = dict_batch_to_device(batch, device)
                y = batch["y"].to(device)
                biases = batch.get("biases", {})
                # y_color = batch["colors"].to(device)
                # y_shape = batch["shapes"].to(device)
                ys.append(y)
                for key, logit in logits.items():
                    # regular
                    correct, total = compute_correct_batch(logit, weights, y, device)
                    key_name = key + "_acc"
                    corrects[key_name] += correct
                    totals[key_name] += total
                    pred = logit.argmax(1)
                    if per_class_precision is not None:
                        prefix = key + "_"
                        per_class_precision[prefix + "precision"].update(pred, y)
                        per_class_recall[prefix + "recall"].update(pred, y)
                    if key in preds:
                        preds[key] = torch.cat((preds[key], pred))
                    else:
                        preds[key] = pred

                    for bias_name in biases:
                        correct, total = compute_correct_batch(logit, weights, biases[bias_name], device)
                        key_name = key + "_acc"
                        corrects_biases[bias_name][key_name] += correct

        for key in corrects:
            results[key] = corrects[key] / totals[key]

        for bias_name in biases:
            for key in corrects_biases[bias_name]:
                results_bias[key + f"_bias_{bias_name}"] = corrects_biases[bias_name][key] / totals[key]

        ##  Best accuracy
        # print([key for key in results if "classif" in key and "acc" in key])
        best_acc = max(results[key] for key in results if "classif" in key and "acc" in key)
        # print([key for key in results if "classif" in key and "acc" in key])
        # print([results[key] for key in results if "classif" in key and "acc" in key])
        results['acc'] = best_acc
        results['average_acc'] = mean([results[key] for key in results if "classif" in key and "acc" in key])

        if per_class_precision is not None:
            for key in per_class_precision:
                precisions = per_class_precision[key].compute()
                for i in range(len(precisions)):
                    name = i
                    if class_names is not None:
                        name = class_names[i]
                    results[key + f"_class-{name}"] = precisions[i].item()
                results[key + f"_class-average"] = precisions.mean().item()
            for key in per_class_recall:
                recalls = per_class_recall[key].compute()
                for i in range(len(recalls)):
                    name = i
                    if class_names is not None:
                        name = class_names[i]
                    results[key + f"_class-{name}"] = recalls[i].item()
                results[key + "_class-average"] = recalls.mean().item()

        if len(preds) > 1:
            all_ratios = []
            ys = torch.cat(ys)
            keys = [p for p in preds if "classif" in p]  #p not in ["ensemble", "main"]]
            for comb in itertools.combinations(keys, 2):
                key1, key2 = comb
                y1 = preds[key1]
                y2 = preds[key2]
                ratio = ratio_errors(ys.cpu().numpy(), y1.cpu().numpy(), y2.cpu().numpy())
                all_ratios.append(ratio)
                if len(keys) != 2:
                    results[f"Ratio/ratio_{key1}_{key2}"] = ratio
                # results[f"Ratio/ratio_{key1}_{key2}"] = ratio
            if all_ratios:
                mean_ratio = mean(all_ratios)
                results["Diversity/mean"] = mean_ratio
        network.train()

        results.update(results_bias)
        return results


class TwoModelsCMNIST(Algorithm):
    CUSTOM_FORWARD = True

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier0 = extend(networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams["nonlinear_classifier"],
                    hparams=self.hparams,
        ))
        self.classifier1 = extend(networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams["nonlinear_classifier"],
                    hparams=self.hparams,
        ))


        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.backpack_loss = extend(nn.CrossEntropyLoss())

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([b["x"] for b in minibatches])
        all_y = torch.cat([b["y"] for b in minibatches])

        biases = {key: torch.cat([b["biases"][key] for b in minibatches]) for key in minibatches[0]["biases"]}

        gt_kernel = (all_y[None, :] == all_y[:, None]).float()
        gt_kernel_biases = {key: (biases[key][None, :] == biases[key][:, None]).float() for key in biases}
        gt_kernel_color = gt_kernel_biases["color"]
        gt_kernel_shape = gt_kernel_biases["shape"]

        features =  self.featurizer(all_x)

        if self.hparams["detach_shape_features"]:
            features2 = features.detach()
        else:
            features2 = features

        logits1 = self.classifier0(features)
        logits2 = self.classifier1(features2)
        loss = 0

        if self.hparams["classifier1"] == "color":
            target1 = biases["color"].long()
        elif self.hparams["classifier1"] == "original":
            target1 = all_y

        if self.hparams["classifier2"] == "shape":
            target2 = biases["shape"].long()
        elif self.hparams["classifier2"] == "original":
            target2 = all_y

        if self.hparams["supervise_logits"]:
            loss1 = F.cross_entropy(logits1, target1)
            loss2 = F.cross_entropy(logits2, target2)
            loss = loss + self.hparams["weight_regular_loss"] * (loss1 + loss2)
        else:
            loss, loss1, loss2 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        # Compute kernel
        if self.hparams["supervise_kernels"]:
            if self.hparams["classifier1"] == "color":
                kernel_target1 = gt_kernel_color
            elif self.hparams["classifier1"] == "original":
                kernel_target1 = gt_kernel

            if self.hparams["classifier2"] == "shape":
                kernel_target2 = gt_kernel_shape
            elif self.hparams["classifier2"] == "original":
                kernel_target2 = gt_kernel

            backpack_loss = self.backpack_loss(logits1, target1) + self.backpack_loss(logits2, target2)
            kernels = kernel_similarity.compute_kernels(backpack_loss,
                            [self.classifier0, self.classifier1],
                            bsize=len(all_x),
                            all_y=all_y,
                            center_gradients=self.hparams["center_gradients"],
                            normalize_gradients=self.hparams["normalize_gradients"],
            )

            if self.hparams["kernel_loss"] == "mse":
                loss_kernel_1 = F.mse_loss(kernels[0], kernel_target1)
                loss_kernel_2 =  F.mse_loss(kernels[1], kernel_target2)
            elif self.hparams["kernel_loss"] == "cos":
                loss_kernel_1 = torch.abs(F.cosine_similarity(kernels[0].flatten(), kernel_target1.flatten(), dim=0))
                loss_kernel_2 =  torch.abs(F.cosine_similarity(kernels[1].flatten(), kernel_target2.flatten(), dim=0))

            loss_kernels = loss_kernel_1 + loss_kernel_2
            loss = loss + self.hparams["weight_kernel_loss"] * loss_kernels
        else:
            loss_kernels = torch.tensor(0.0)

        # similarity_matrix = kernel_similarity.batch_similarity(kernels, "cos", "abs")
        # assert similarity_matrix.shape == (2, 2)
        # Kernel losses ?

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        out = {
            "loss": loss.item(),
            "loss1": loss1.item(),
            "loss2": loss2.item(),
            "loss_kernels": loss_kernels.item(),
            "loss_kernel_1":  loss_kernel_1.item(),
            "loss_kernel_2": loss_kernel_2.item(),
        }

        # for i in range(len(distances_with_gt)):
        #     out[f"sim_with_gt_classif-{i}"] = distances_with_gt[i].item()
        #     for key in biases:
        #         out[f"sim_with_gt_{key}_classif-{i}"] = distances_with_gt_biases[key][i].item()
        return out

    def predict(self, batch):
        feats = self.featurizer(batch["x"].cuda())
        results = {
            "classif-0": self.classifier0(feats),
            "classif-1": self.classifier1(feats)
        }
        return results
        logits = [classif(feats) for classif in self.classifiers]
        true_logits = self.true_logits_classifier(feats)
        ensemble_logits = sum(logits)
        results = {
            f"classif-{i}": logits[i] for i in range(len(logits))
        }
        results["ensemble"] = ensemble_logits
        results["pred_true_logits"] = true_logits
        return results

    @staticmethod
    def accuracy(network, loader, weights, device, num_classes=None, class_names=None):
        network.eval()
        corrects = defaultdict(int)  # key -> int
        corrects_biases = defaultdict(lambda: defaultdict(int))
        # corrects_color = defaultdict(int)
        # corrects_shape = defau
        totals = defaultdict(int)    # key -> int
        results = dict()
        preds =  dict()
        ys = []
        results_bias = dict()

        # if Precision is not None and num_classes is not None:
        #     per_class_precision = defaultdict(lambda: Precision(num_classes=num_classes, average="none").to(device))
        #     per_class_recall =  defaultdict(lambda: Recall(num_classes=num_classes, average="none").to(device))
        # else:
        per_class_precision, per_class_recall = None, None

        with torch.no_grad():
            for batch in loader:
                logits = network.predict(batch)
                batch = dict_batch_to_device(batch, device)
                y = batch["y"].to(device)
                biases = batch.get("biases", {})
                # y_color = batch["colors"].to(device)
                # y_shape = batch["shapes"].to(device)
                ys.append(y)
                for key, logit in logits.items():
                    # regular
                    correct, total = compute_correct_batch(logit, weights, y, device)
                    key_name = key + "_acc"
                    corrects[key_name] += correct
                    totals[key_name] += total
                    pred = logit.argmax(1)
                    if per_class_precision is not None:
                        prefix = key + "_"
                        per_class_precision[prefix + "precision"].update(pred, y)
                        per_class_recall[prefix + "recall"].update(pred, y)
                    if key in preds:
                        preds[key] = torch.cat((preds[key], pred))
                    else:
                        preds[key] = pred

                    for bias_name in biases:
                        correct, total = compute_correct_batch(logit, weights, biases[bias_name], device)
                        key_name = key + "_acc"
                        corrects_biases[bias_name][key_name] += correct

        for key in corrects:
            results[key] = corrects[key] / totals[key]

        for bias_name in biases:
            for key in corrects_biases[bias_name]:
                results_bias[key + f"_bias_{bias_name}"] = corrects_biases[bias_name][key] / totals[key]

        ##  Best accuracy
        # print([key for key in results if "classif" in key and "acc" in key])
        best_acc = max(results[key] for key in results if "classif" in key and "acc" in key)
        # print([key for key in results if "classif" in key and "acc" in key])
        # print([results[key] for key in results if "classif" in key and "acc" in key])
        results['acc'] = best_acc
        results['average_acc'] = mean([results[key] for key in results if "classif" in key and "acc" in key])

        if per_class_precision is not None:
            for key in per_class_precision:
                precisions = per_class_precision[key].compute()
                for i in range(len(precisions)):
                    name = i
                    if class_names is not None:
                        name = class_names[i]
                    results[key + f"_class-{name}"] = precisions[i].item()
                results[key + f"_class-average"] = precisions.mean().item()
            for key in per_class_recall:
                recalls = per_class_recall[key].compute()
                for i in range(len(recalls)):
                    name = i
                    if class_names is not None:
                        name = class_names[i]
                    results[key + f"_class-{name}"] = recalls[i].item()
                results[key + "_class-average"] = recalls.mean().item()

        if len(preds) > 1:
            all_ratios = []
            ys = torch.cat(ys)
            keys = [p for p in preds if "classif" in p]  #p not in ["ensemble", "main"]]
            for comb in itertools.combinations(keys, 2):
                key1, key2 = comb
                y1 = preds[key1]
                y2 = preds[key2]
                ratio = ratio_errors(ys.cpu().numpy(), y1.cpu().numpy(), y2.cpu().numpy())
                all_ratios.append(ratio)
                if len(keys) != 2:
                    results[f"Ratio/ratio_{key1}_{key2}"] = ratio
                # results[f"Ratio/ratio_{key1}_{key2}"] = ratio
            if all_ratios:
                mean_ratio = mean(all_ratios)
                results["Diversity/mean"] = mean_ratio
        network.train()

        results.update(results_bias)
        return results
