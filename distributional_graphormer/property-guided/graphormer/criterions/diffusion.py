# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import pickle as pkl

@register_criterion("diffusion_loss", dataclass=FairseqDataclass)
class DiffusionLoss(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        sample_size = sample["nsamples"]

        if model.training:
            output = model.get_training_output(**sample["net_input"])
        else:
            with torch.no_grad():
                output = model.get_sampling_output(**sample["net_input"])

        persample_loss = output["persample_loss"]
        loss = torch.sum(persample_loss)
        sample_size = output["sample_size"]

        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
        }

        if "persample_rmsd" in output:
            persample_rmsd = output["persample_rmsd"]
            rmsd = torch.sum(persample_rmsd)
            logging_output["rmsd"] = rmsd.data

        if "persample_lattice_rmsd" in output:
            persample_lattice_rmsd = output["persample_lattice_rmsd"]
            rmsd = torch.sum(persample_lattice_rmsd)
            logging_output["lattice_rmsd"] = rmsd.data

        if "persample_rmsd_pbc" in output:
            persample_rmsd = output["persample_rmsd_pbc"]
            rmsd_pbc = torch.sum(persample_rmsd)
            logging_output["rmsd_pbc"] = rmsd_pbc.data

        if "persample_pred_pos_rmsd" in output:
            persample_pred_pos_rmsd = output["persample_pred_pos_rmsd"]
            pred_pos_rmsd = torch.sum(persample_pred_pos_rmsd)
            logging_output["pred_pos_rmsd"] = pred_pos_rmsd.data

        if "persample_pred_pos_direct_rmsd" in output:
            persample_pred_pos_direct_rmsd = output["persample_pred_pos_direct_rmsd"]
            pred_pos_direct_rmsd = torch.sum(persample_pred_pos_direct_rmsd)
            logging_output["pred_pos_direct_rmsd"] = pred_pos_direct_rmsd.data

        if "persample_radius_loss" in output:
            persample_radius_loss = torch.sum(output["persample_radius_loss"])
            radius_loss = torch.sum(persample_radius_loss)
            logging_output["diffusion_loss"] = loss.data.clone()
            loss += radius_loss
            logging_output["radius_loss"] = radius_loss.data
            logging_output["loss"] = loss.data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        rmsd_sum = sum(log.get("rmsd", 0) for log in logging_outputs)
        rmsd_pbc_sum = sum(log.get("rmsd_pbc", 0) for log in logging_outputs)
        lattice_rmsd_sum = sum(log.get("lattice_rmsd", 0) for log in logging_outputs)
        pred_pos_rmsd_sum = sum(log.get("pred_pos_rmsd", 0) for log in logging_outputs)
        pred_pos_direct_rmsd_sum = sum(log.get("pred_pos_direct_rmsd", 0) for log in logging_outputs)
        radius_loss_sum = sum(log.get("radius_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        if rmsd_sum > 0:
            metrics.log_scalar("rmsd", rmsd_sum / sample_size, sample_size, round=6)
        if lattice_rmsd_sum > 0:
            metrics.log_scalar("lattice_rmsd", lattice_rmsd_sum / sample_size, sample_size, round=6)
        if rmsd_pbc_sum > 0:
            metrics.log_scalar("rmsd_pbc", rmsd_pbc_sum / sample_size, sample_size, round=6)
        if pred_pos_rmsd_sum > 0:
            metrics.log_scalar("pred_pos_rmsd", pred_pos_rmsd_sum / sample_size, sample_size, round=6)
        if pred_pos_direct_rmsd_sum > 0:
            metrics.log_scalar("pred_pos_direct_rmsd", pred_pos_direct_rmsd_sum / sample_size, sample_size, round=6)
        if radius_loss_sum > 0:
            metrics.log_scalar("radius_loss", radius_loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True


@dataclass
class OCKDEDataclass(FairseqDataclass):
    n_kde_samples: int = field(default=10, metadata={"help": "number of samples per oc system"})
    kernel_func: str = field(default="normal", metadata={"help": "kernel function used in KDE"})
    kde_temperature: float = field(default=0.1, metadata={"help": "temperature of kernel function"})
    result_save_dir: str = field(default="flow_ode_res_save_dir", metadata={"help": "directory to save flow ode results"})


@register_criterion("oc_kde", dataclass=OCKDEDataclass)
class OCKDE(DiffusionLoss):
    def __init__(self, cfg: OCKDEDataclass):
        super().__init__(cfg)
        self.n_kde_samples = cfg.n_kde_samples
        self.kernel_func = cfg.kernel_func
        self.kde_temperature = cfg.kde_temperature
        self.result_save_dir = cfg.result_save_dir

    def calc_kde_probs_from_rmsd(
        self,
        rmsd
    ):
        all_probs = 1.0 / np.sqrt(2.0 * 3.1415926) * torch.exp(-(rmsd ** 2) / self.kde_temperature) # B x N1 x N2
        probs = torch.mean(all_probs, axis=-1) / self.kde_temperature # B x N1
        probs /= (torch.sum(probs, axis=-1, keepdim=True) + 1e-32)
        return probs

    def forward(self, model, sample, reduce=True):
        if model.training:
            return super().forward(model, sample, reduce)
        else:
            sample_size = sample["nsamples"]
            batched_data = sample["net_input"]["batched_data"]
            if "all_poses" not in batched_data:
                return super().forward(model, sample, reduce)
            else:
                num_moveable = batched_data["num_moveable"]
                all_poses = batched_data["all_poses"]
                all_outputs = []
                ori_pos = batched_data["pos"].clone()
                all_losses = 0.0
                for _ in tqdm(range(self.n_kde_samples)):
                    with torch.no_grad():
                        batched_data["pos"] = ori_pos.clone()
                        output = model.get_sampling_output(batched_data)
                        persample_loss = output["persample_loss"]
                        all_losses += torch.sum(persample_loss)
                        sample_size = output["sample_size"]
                        pred_pos = output["pred_pos"] # B x T x 3
                    all_outputs.append(pred_pos.clone().unsqueeze(-3))
                all_outputs = torch.cat(all_outputs, axis=-3) # B x N1 x T x 3
                device = str(num_moveable.device).replace(':', '_')
                with open(f"{self.result_save_dir}/{device}_pos.out", "ab") as out_file:
                    sids = batched_data["sid"]
                    atoms = batched_data["x"][:, :, 0]
                    tags = batched_data["tags"]
                    pkl.dump((sids, atoms, tags, all_outputs, all_poses), out_file)
                all_losses /= self.n_kde_samples

                atoms = batched_data["x"][:, :, 0]
                mask = (~atoms.eq(0)).unsqueeze(1).unsqueeze(2).float()

                rmsds = torch.sqrt(torch.sum(torch.sum((all_outputs.unsqueeze(1) - all_outputs.unsqueeze(2)) ** 2, dim=-1) * mask, dim=-1) / num_moveable.unsqueeze(1).unsqueeze(2))

                rmsds_ref = torch.sqrt(torch.sum(torch.sum((all_outputs.unsqueeze(2) - all_poses.unsqueeze(1)) ** 2, dim=-1) * mask, dim=-1) / num_moveable.unsqueeze(1).unsqueeze(2))

                mean_output = torch.mean(all_outputs, dim=1) # B x T x 3
                rmsd_of_mean_output = torch.mean(torch.sqrt(torch.sum(torch.sum((mean_output.unsqueeze(1) - all_poses) ** 2, dim=-1) * (~atoms.eq(0)).unsqueeze(1).float(), dim=-1) / num_moveable.unsqueeze(1)), dim=1) # B

                self_probs = self.calc_kde_probs_from_rmsd(rmsds) # B x N1
                ref_probs = self.calc_kde_probs_from_rmsd(rmsds_ref) # B x N1
                kl_div = torch.sum(self_probs * torch.log(self_probs / (ref_probs + 1e-32)), axis=1) / (torch.sum(self_probs, dim=1) + 1e-32)
                kl_div_sum = torch.sum(kl_div)
                logging_output = {
                    "loss": all_losses.data,
                    "kde_kl": kl_div_sum.data,
                    "sample_size": sample_size,
                    "nsentences": sample_size,
                }

                logging_output["rmsd"] = torch.sum(torch.mean(rmsds_ref, dim=[1, 2])).data
                logging_output["rmsd_of_mean_output"] = torch.sum(rmsd_of_mean_output).data
                logging_output["min_rmsd"] = torch.sum(torch.mean(torch.min(rmsds_ref, dim=-1)[0], dim=-1)).data

                return kl_div_sum, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        rmsd_of_mean_output_sum = sum(log.get("rmsd_of_mean_output", 0) for log in logging_outputs)
        rmsd_sum = sum(log.get("rmsd", 0) for log in logging_outputs)
        radius_loss_sum = sum(log.get("radius_loss", 0) for log in logging_outputs)
        kde_kl_sum = sum(log.get("kde_kl", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        min_rmsd_sum = sum(log.get("min_rmsd", 0) for log in logging_outputs)
        diffusion_loss_sum = sum(log.get("diffusion_loss", 0) for log in logging_outputs)

        metrics.log_scalar("diffusion_loss", diffusion_loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("rmsd", rmsd_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("rmsd_of_mean_output", rmsd_of_mean_output_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("kde_kl", kde_kl_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("min_rmsd", min_rmsd_sum / sample_size, sample_size, round=6)
        if radius_loss_sum > 0:
            metrics.log_scalar("radius_loss", radius_loss_sum / sample_size, sample_size, round=6)



@register_criterion("flow_ode", dataclass=OCKDEDataclass)
class FlowODE(DiffusionLoss):
    def __init__(self, cfg: OCKDEDataclass):
        super().__init__(cfg)
        self.n_kde_samples = cfg.n_kde_samples
        self.kernel_func = cfg.kernel_func
        self.kde_temperature = cfg.kde_temperature
        self.result_save_dir = cfg.result_save_dir

    def calc_kde_probs_from_rmsd(
        self,
        rmsd
    ):
        all_probs = 1.0 / np.sqrt(2.0 * 3.1415926) * torch.exp(-(rmsd ** 2) / self.kde_temperature) # B x N1 x N2
        probs = torch.mean(all_probs, axis=-1) / self.kde_temperature # B x N1
        probs /= (torch.sum(probs, axis=-1, keepdim=True) + 1e-32)
        return probs

    def forward(self, model, sample, reduce=True):
        if model.training:
            return super().forward(model, sample, reduce)
        else:
            sample_size = sample["nsamples"]
            batched_data = sample["net_input"]["batched_data"]
            if "all_poses" not in batched_data:
                return super().forward(model, sample, reduce)
            else:
                num_moveable = batched_data["num_moveable"]
                all_poses = batched_data["all_poses"]
                all_outputs = []
                all_likelihood = []
                all_prior_log_p = []
                all_pred_pos = []
                all_latent_pos = []
                ori_pos = batched_data["pos"].clone()
                for _ in tqdm(range(self.n_kde_samples)):
                    all_losses = 0.0
                    with torch.no_grad():
                        batched_data["pos"] = ori_pos.clone()
                        output = model.get_sampling_output(batched_data)
                        persample_loss = output["persample_loss"]
                        all_losses += torch.sum(persample_loss)
                        sample_size = output["sample_size"]
                        pred_pos = output["pred_pos"] # B x T x 3
                        all_pred_pos.append(pred_pos.clone().unsqueeze(1))
                        batched_data["pos"] = ori_pos.clone()
                        batched_data["pred_pos"] = pred_pos
                        flow_ode_output = model.get_flow_ode_output(batched_data)
                        all_likelihood.append(flow_ode_output["persample_likelihood"].unsqueeze(1))
                        all_prior_log_p.append(flow_ode_output["persample_prior_log_p"].unsqueeze(1))
                        all_latent_pos.append(flow_ode_output["latent_pos"].unsqueeze(1))
                        device = str(num_moveable.device).replace(':', '_')
                        with open(f"{self.result_save_dir}/{device}.out", "ab") as out_file:
                            sids = batched_data["sid"]
                            pkl.dump((sids, all_likelihood[-1], all_prior_log_p[-1], all_pred_pos[-1], all_latent_pos[-1]), out_file)
                        all_losses /= self.n_kde_samples
                        all_outputs.append(pred_pos.clone().unsqueeze(-3))
                all_outputs = torch.cat(all_outputs, axis=-3) # B x N1 x T x 3
                all_likelihood = torch.cat(all_likelihood, axis=-1)
                all_prior_log_p = torch.cat(all_prior_log_p, axis=-1)
                all_likelihood_max, _ = torch.max(all_likelihood, axis=1)
                all_pred_pos = torch.cat(all_pred_pos, dim=1)
                all_latent_pos = torch.cat(all_latent_pos, dim=1)
                self_probs = torch.exp(all_likelihood - all_likelihood_max.unsqueeze(1))
                self_probs /= torch.sum(self_probs, dim=-1, keepdim=True)

                atoms = batched_data["x"][:, :, 0]
                mask = (~atoms.eq(0)).unsqueeze(1).unsqueeze(2).float()

                rmsds = torch.sqrt(torch.sum(torch.sum((all_outputs.unsqueeze(1) - all_outputs.unsqueeze(2)) ** 2, dim=-1) * mask, dim=-1) / num_moveable.unsqueeze(1).unsqueeze(2))
                rmsds_ref = torch.sqrt(torch.sum(torch.sum((all_outputs.unsqueeze(2) - all_poses.unsqueeze(1)) ** 2, dim=-1) * mask, dim=-1) / num_moveable.unsqueeze(1).unsqueeze(2))

                mean_output = torch.mean(all_outputs, dim=1) # B x T x 3
                rmsd_of_mean_output = torch.mean(torch.sqrt(torch.sum(torch.sum((mean_output.unsqueeze(1) - all_poses) ** 2, dim=-1) * (~atoms.eq(0)).unsqueeze(1).float(), dim=-1) / num_moveable.unsqueeze(1)), dim=1) # B

                rmsd_self_probs = self.calc_kde_probs_from_rmsd(rmsds) # B x N1
                ref_probs = self.calc_kde_probs_from_rmsd(rmsds_ref) # B x N1
                kl_div = torch.sum(self_probs * torch.log(self_probs / (ref_probs + 1e-32)), axis=1) / (torch.sum(self_probs, dim=1) + 1e-32)
                rmsd_kl_div = torch.sum(rmsd_self_probs * torch.log(rmsd_self_probs / (ref_probs + 1e-32)), axis=1) / (torch.sum(rmsd_self_probs, dim=1) + 1e-32)
                kl_div_sum = torch.sum(kl_div)
                rmsd_kl_div_sum = torch.sum(rmsd_kl_div)
                logging_output = {
                    "loss": all_losses.data,
                    "kde_kl": kl_div_sum.data,
                    "rmsd_kde_kl": rmsd_kl_div_sum.data,
                    "sample_size": sample_size,
                    "nsentences": sample_size,
                }

                logging_output["rmsd"] = torch.sum(torch.mean(rmsds_ref, dim=[1, 2])).data
                logging_output["rmsd_of_mean_output"] = torch.sum(rmsd_of_mean_output).data
                logging_output["min_rmsd"] = torch.sum(torch.mean(torch.min(rmsds_ref, dim=-1)[0], dim=-1)).data

                return kl_div_sum, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        rmsd_of_mean_output_sum = sum(log.get("rmsd_of_mean_output", 0) for log in logging_outputs)
        rmsd_sum = sum(log.get("rmsd", 0) for log in logging_outputs)
        kde_kl_sum = sum(log.get("kde_kl", 0) for log in logging_outputs)
        rmsd_kde_kl_sum = sum(log.get("rmsd_kde_kl", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        min_rmsd_sum = sum(log.get("min_rmsd", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("rmsd", rmsd_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("rmsd_of_mean_output", rmsd_of_mean_output_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("kde_kl", kde_kl_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("rmsd_kde_kl", rmsd_kde_kl_sum / sample_size, sample_size, round=6)
        metrics.log_scalar("min_rmsd", min_rmsd_sum / sample_size, sample_size, round=6)
