"""Generic SNN architecture parameterised by neuron type.

One module that covers all nine snntorch neuron families. Wiring differs by
neuron family:

  * SIMPLE  (Leaky, Lapicque, Synaptic, Alpha):
        Linear → Neuron → Linear → Neuron_out, looped over T timesteps.
  * RECURRENT (RLeaky, RSynaptic):
        Linear → RNeuron(hidden→hidden recurrent) → Linear → Leaky_out, looped.
  * SLSTM:
        SLSTM(F→H) consumed T-step by T-step → Linear → Leaky_out.
  * SConv2dLSTM:
        Reshape F→1×H×W, SConv2dLSTM → flatten → Linear → Leaky_out (T-step loop).
  * LeakyParallel:
        Vectorised across T (no python loop inside), → Linear → Leaky_out.

Output: spike-count tensor of shape [B, num_classes] — used as logits for CE.
Also exposes `total_spikes` for the energy proxy metric.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn_lib

from src.neurons import (
    SIMPLE_NEURONS, RECURRENT_NEURONS, SEQUENCE_NEURONS,
    get_neuron_factory, DEFAULT_BETA,
)


class SNN(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_classes: int,
                 neuron_name: str, num_steps: int, conv_out_channels: int = 8):
        super().__init__()
        self.neuron_name = neuron_name
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_features = num_features
        self.conv_out_channels = conv_out_channels
        self._build()

    # ------------- construction -------------
    def _build(self):
        n = self.neuron_name
        if n in SIMPLE_NEURONS:
            self.fc1 = nn.Linear(self.num_features, self.hidden_size)
            self.lif1 = get_neuron_factory(n)()
            self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
            self.lif2 = get_neuron_factory(n)()
        elif n in RECURRENT_NEURONS:
            self.fc1 = nn.Linear(self.num_features, self.hidden_size)
            self.rlif = get_neuron_factory(n)(self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        elif n == "SLSTM":
            self.slstm = get_neuron_factory(n)(self.num_features, self.hidden_size)
            self.fc = nn.Linear(self.hidden_size, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        elif n == "SConv2dLSTM":
            side = math.ceil(math.sqrt(self.num_features))
            self.spatial_side = side
            self.padded_features = side * side
            self.sconv = get_neuron_factory(n)(in_channels=1,
                                               out_channels=self.conv_out_channels,
                                               kernel_size=3)
            flat = self.conv_out_channels * side * side
            self.fc = nn.Linear(flat, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        elif n == "LeakyParallel":
            self.lp = get_neuron_factory(n)(self.num_features, self.hidden_size)
            self.fc = nn.Linear(self.hidden_size, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        else:
            raise ValueError(f"Unsupported neuron {n}")

    # ------------- introspection -------------
    def _spike_sites(self, T: int, B: int) -> int:
        """Number of binary spike outputs the forward pass produces — denominator
        for the sparsity metric. Output-layer + hidden-layer counts only."""
        n = self.neuron_name
        if n == "SConv2dLSTM":
            hidden = self.conv_out_channels * self.spatial_side * self.spatial_side
        else:
            hidden = self.hidden_size
        return T * B * (hidden + self.num_classes)

    def synaptic_fanout(self) -> dict:
        """Synapses driven by one spike at each layer, for the SOP proxy.

            SOP = input_spikes x fanout_input + hidden_spikes x fanout_hidden

        An accumulate happens once per (spike, downstream synapse), so the
        fanout of a layer is the width of the projection its spikes feed.

        Two conventions are recorded rather than assumed, because they are the
        kind of thing a reader has to be able to check:

        * **Gated cells** (SLSTM, SConv2dLSTM) drive four gate projections per
          input, so their input fanout is 4x the hidden width. Counting one
          would understate their cost by 4x and flatter them in the comparison.
        * **SConv2dLSTM** is convolutional: one input spike reaches
          ``out_channels * k^2`` synapses, independent of the spatial size, and
          again 4x for the gates.

        The output layer's spikes drive nothing downstream -- they are the
        readout -- so they contribute no SOPs and appear only in the spike
        decomposition.
        """
        n = self.neuron_name
        if n == "SConv2dLSTM":
            k = 3
            return {
                "fanout_input": 4 * self.conv_out_channels * k * k,
                "fanout_hidden": self.num_classes,
                "hidden_width": self.conv_out_channels * self.spatial_side ** 2,
                "gated": True,
            }
        if n == "SLSTM":
            return {"fanout_input": 4 * self.hidden_size,
                    "fanout_hidden": self.num_classes,
                    "hidden_width": self.hidden_size, "gated": True}
        return {"fanout_input": self.hidden_size,
                "fanout_hidden": self.num_classes,
                "hidden_width": self.hidden_size, "gated": False}

    # ------------- forward -------------
    def forward(self, spike_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """spike_train: [T, B, F].

        Returns
        -------
        logits        : [B, num_classes] — spike-count at output layer.
        total_spikes  : scalar tensor    — sum of all hidden+output spikes (energy proxy).
        spike_sites   : int              — number of (T × neuron × sample) sites that
                                            *could* spike; used for sparsity.
        """
        T, B, F = spike_train.shape
        n = self.neuron_name
        device = spike_train.device
        spike_sites = self._spike_sites(T, B)

        if n in SIMPLE_NEURONS:
            # Initialise per-neuron state vectors.
            if n == "Leaky":
                mem1 = self.lif1.init_leaky()
                mem2 = self.lif2.init_leaky()
            elif n == "Lapicque":
                mem1 = self.lif1.init_lapicque()
                mem2 = self.lif2.init_lapicque()
            elif n == "Synaptic":
                syn1, mem1 = self.lif1.init_synaptic()
                syn2, mem2 = self.lif2.init_synaptic()
            elif n == "Alpha":
                syn_e1, syn_i1, mem1 = self.lif1.init_alpha()
                syn_e2, syn_i2, mem2 = self.lif2.init_alpha()
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                cur1 = self.fc1(spike_train[t])
                if n in {"Leaky", "Lapicque"}:
                    spk1, mem1 = self.lif1(cur1, mem1)
                elif n == "Synaptic":
                    spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
                else:  # Alpha
                    spk1, syn_e1, syn_i1, mem1 = self.lif1(cur1, syn_e1, syn_i1, mem1)
                cur2 = self.fc2(spk1)
                if n in {"Leaky", "Lapicque"}:
                    spk2, mem2 = self.lif2(cur2, mem2)
                elif n == "Synaptic":
                    spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                else:  # Alpha
                    spk2, syn_e2, syn_i2, mem2 = self.lif2(cur2, syn_e2, syn_i2, mem2)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            spikes_out = torch.stack(spk_out_record, dim=0)  # [T, B, C]
            return spikes_out.sum(dim=0), spike_total, spike_sites

        if n in RECURRENT_NEURONS:
            init = self.rlif.init_rleaky() if n == "RLeaky" else self.rlif.init_rsynaptic()
            if n == "RLeaky":
                spk1, mem1 = init
                syn1 = None
            else:
                spk1, syn1, mem1 = init
            spk1 = spk1.to(device); mem1 = mem1.to(device)
            if syn1 is not None:
                syn1 = syn1.to(device)
            # Make sure shape matches batch
            if spk1.shape[0] != B:
                spk1 = torch.zeros(B, self.hidden_size, device=device)
                mem1 = torch.zeros(B, self.hidden_size, device=device)
                if syn1 is not None:
                    syn1 = torch.zeros(B, self.hidden_size, device=device)
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                cur1 = self.fc1(spike_train[t])
                if n == "RLeaky":
                    spk1, mem1 = self.rlif(cur1, spk1, mem1)
                else:
                    spk1, syn1, mem1 = self.rlif(cur1, spk1, syn1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem_out = self.lif_out(cur2, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        if n == "SLSTM":
            syn, mem = self.slstm.init_slstm()
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                spk1, syn, mem = self.slstm(spike_train[t], syn, mem)
                cur = self.fc(spk1)
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        if n == "SConv2dLSTM":
            syn, mem = self.sconv.init_sconv2dlstm()
            mem_out = self.lif_out.init_leaky()
            pad = self.padded_features - F
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                x = spike_train[t]
                if pad > 0:
                    x = torch.cat([x, torch.zeros(B, pad, device=device, dtype=x.dtype)], dim=1)
                x = x.view(B, 1, self.spatial_side, self.spatial_side)
                spk1, syn, mem = self.sconv(x, syn, mem)
                flat = spk1.flatten(1)
                cur = self.fc(flat)
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        if n == "LeakyParallel":
            spikes = self.lp(spike_train)  # [T, B, H]
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            spike_total = spikes.sum()
            for t in range(T):
                cur = self.fc(spikes[t])
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        raise ValueError(f"Unsupported neuron {n}")  # pragma: no cover

    # ------------- early-exit + SynOps (additive; forward above is unchanged) -------------
    def _output_spike_train(self, spike_train: torch.Tensor):
        """Re-runs the per-family loop but returns the *per-timestep* output-spike
        stack ``[T, B, C]`` plus the total spikes, the pre-readout (hidden) spike
        count, and spike sites. Used only by ``predict_early_exit`` — ``forward``
        is left byte-for-byte unchanged for reproducibility. The branch bodies
        mirror ``forward`` exactly; the only addition is tracking ``hidden_total``
        (spikes feeding the final classification layer) for the SynOps proxy.
        """
        T, B, F = spike_train.shape
        n = self.neuron_name
        device = spike_train.device
        spike_sites = self._spike_sites(T, B)
        hidden_total = torch.zeros((), device=device)
        spike_total = torch.zeros((), device=device)

        if n in SIMPLE_NEURONS:
            if n == "Leaky":
                mem1 = self.lif1.init_leaky(); mem2 = self.lif2.init_leaky()
            elif n == "Lapicque":
                mem1 = self.lif1.init_lapicque(); mem2 = self.lif2.init_lapicque()
            elif n == "Synaptic":
                syn1, mem1 = self.lif1.init_synaptic(); syn2, mem2 = self.lif2.init_synaptic()
            else:  # Alpha
                syn_e1, syn_i1, mem1 = self.lif1.init_alpha()
                syn_e2, syn_i2, mem2 = self.lif2.init_alpha()
            spk_out_record = []
            for t in range(T):
                cur1 = self.fc1(spike_train[t])
                if n in {"Leaky", "Lapicque"}:
                    spk1, mem1 = self.lif1(cur1, mem1)
                elif n == "Synaptic":
                    spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
                else:
                    spk1, syn_e1, syn_i1, mem1 = self.lif1(cur1, syn_e1, syn_i1, mem1)
                cur2 = self.fc2(spk1)
                if n in {"Leaky", "Lapicque"}:
                    spk2, mem2 = self.lif2(cur2, mem2)
                elif n == "Synaptic":
                    spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                else:
                    spk2, syn_e2, syn_i2, mem2 = self.lif2(cur2, syn_e2, syn_i2, mem2)
                spk_out_record.append(spk2)
                hidden_total = hidden_total + spk1.sum()
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record, 0), spike_total, hidden_total, spike_sites

        if n in RECURRENT_NEURONS:
            init = self.rlif.init_rleaky() if n == "RLeaky" else self.rlif.init_rsynaptic()
            if n == "RLeaky":
                spk1, mem1 = init; syn1 = None
            else:
                spk1, syn1, mem1 = init
            spk1 = spk1.to(device); mem1 = mem1.to(device)
            if syn1 is not None:
                syn1 = syn1.to(device)
            if spk1.shape[0] != B:
                spk1 = torch.zeros(B, self.hidden_size, device=device)
                mem1 = torch.zeros(B, self.hidden_size, device=device)
                if syn1 is not None:
                    syn1 = torch.zeros(B, self.hidden_size, device=device)
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            for t in range(T):
                cur1 = self.fc1(spike_train[t])
                if n == "RLeaky":
                    spk1, mem1 = self.rlif(cur1, spk1, mem1)
                else:
                    spk1, syn1, mem1 = self.rlif(cur1, spk1, syn1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem_out = self.lif_out(cur2, mem_out)
                spk_out_record.append(spk2)
                hidden_total = hidden_total + spk1.sum()
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record, 0), spike_total, hidden_total, spike_sites

        if n == "SLSTM":
            syn, mem = self.slstm.init_slstm()
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            for t in range(T):
                spk1, syn, mem = self.slstm(spike_train[t], syn, mem)
                cur = self.fc(spk1)
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                hidden_total = hidden_total + spk1.sum()
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record, 0), spike_total, hidden_total, spike_sites

        if n == "SConv2dLSTM":
            syn, mem = self.sconv.init_sconv2dlstm()
            mem_out = self.lif_out.init_leaky()
            pad = self.padded_features - F
            spk_out_record = []
            for t in range(T):
                x = spike_train[t]
                if pad > 0:
                    x = torch.cat([x, torch.zeros(B, pad, device=device, dtype=x.dtype)], dim=1)
                x = x.view(B, 1, self.spatial_side, self.spatial_side)
                spk1, syn, mem = self.sconv(x, syn, mem)
                flat = spk1.flatten(1)
                cur = self.fc(flat)
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                hidden_total = hidden_total + spk1.sum()
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record, 0), spike_total, hidden_total, spike_sites

        if n == "LeakyParallel":
            spikes = self.lp(spike_train)  # [T, B, H]
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            hidden_total = spikes.sum()
            spike_total = spikes.sum()
            for t in range(T):
                cur = self.fc(spikes[t])
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk2.sum()
            return torch.stack(spk_out_record, 0), spike_total, hidden_total, spike_sites

        raise ValueError(f"Unsupported neuron {n}")  # pragma: no cover

    @torch.no_grad()
    def predict_early_exit(self, spike_train: torch.Tensor,
                           theta_exit: float = 0.9, theta_uncertain: float = 0.5):
        """Confidence-triggered early exit over the T latency steps, plus a SynOps
        proxy. The running readout is the cumulative output-spike *rate*
        (cumsum / t), softmaxed; a sample exits when its confidence first crosses
        ``theta_exit``. Returns (preds [B], exit_step [B], uncertain [B],
        total_spikes, spike_sites, synops_per_sample).
        """
        stack, total, hidden, sites = self._output_spike_train(spike_train)  # stack [T,B,C]
        T, B, C = stack.shape
        # Running readout = cumulative output-spike *count* (NOT a rate): at t=T
        # this equals the forward logits, so the confidence is on the same scale
        # as the final decision and sharpens as spike evidence accumulates. A
        # rate (count/t) would stay in [0,1] and its softmax would be too flat to
        # ever cross the exit threshold.
        running = torch.cumsum(stack, dim=0)                       # [T, B, C] cumulative counts
        conf, pred = torch.softmax(running, dim=-1).max(dim=-1)    # [T, B]
        exit_step = torch.full((B,), T, dtype=torch.long, device=stack.device)
        done = torch.zeros(B, dtype=torch.bool, device=stack.device)
        preds = torch.zeros(B, dtype=torch.long, device=stack.device)
        for t in range(T):
            fire = (~done) & (conf[t] >= theta_exit)
            preds[fire] = pred[t][fire]; exit_step[fire] = t + 1; done = done | fire
        preds[~done] = pred[T - 1][~done]
        uncertain = (~done) & (conf[T - 1] < theta_uncertain)
        # SynOps proxy: TOTAL synaptic accumulate ops summed over both synaptic
        # layers (so it is comparable to an iso-architecture dense-ANN MAC count,
        # which also counts all layers):
        #   layer 1 (input -> hidden): input_spikes  x  hidden fan-out
        #   layer 2 (hidden -> class): hidden_spikes  x  num_classes fan-out
        # For SConv2dLSTM the layer-1 fan-out uses hidden_size as an approximation
        # of the convolutional fan-out (documented caveat).
        input_spikes = spike_train.sum()
        synops = (input_spikes * self.hidden_size + hidden * self.num_classes) / B
        return preds, exit_step, uncertain, total, sites, synops
