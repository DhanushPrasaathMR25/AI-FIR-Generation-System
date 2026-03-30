# ---------------------------------------------
# FORCE FULL OFFLINE MODE (NO INTERNET CALLS)
# ---------------------------------------------
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ---------------------------------------------
# IMPORTS
# ---------------------------------------------
import math
import torch
import torch.nn as nn
import pennylane as qml
from transformers import AutoModel, AutoConfig, logging

# Silence all transformer logs
logging.set_verbosity_error()

# ---------------------------------------------
# CONSTANTS
# ---------------------------------------------
INLEGAL = "law-ai/InLegalBERT"
NUM_CAT = 44
NUM_SEC = 45


# ---------------------------------------------
# QUANTUM LAYER
# ---------------------------------------------
def create_qml_torch_layer(n_qubits=4, n_layers=1, output_dim=None, dev_name="default.qubit"):
    dev = qml.device(dev_name, wires=n_qubits)

    weight_shape = (n_layers, n_qubits, 3)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    qlayer = qml.qnn.TorchLayer(qnode, {"weights": weight_shape})

    class QModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            out_dim = n_qubits if output_dim is None else output_dim
            self.post = nn.Sequential(
                nn.Linear(n_qubits, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim)
            )

        def forward(self, x):
            q_out = self.qlayer(x)
            target_device = self.post[0].weight.device
            q_out = q_out.to(target_device)
            return self.post(q_out)

    return QModule()


# ---------------------------------------------
# HYBRID MODEL
# ---------------------------------------------
class HybridInLegal(nn.Module):

    def __init__(self, transformer_name=INLEGAL, n_qubits=4, q_out_dim=64, quantum_on=True):
        super().__init__()

        # Load config (offline)
        cfg = AutoConfig.from_pretrained(
            transformer_name,
            local_files_only=True
        )

        # Load backbone STRICTLY from local cache
        self.transformer = AutoModel.from_pretrained(
            transformer_name,
            config=cfg,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )

        self.quantum_on = quantum_on
        hidden_size = self.transformer.config.hidden_size

        # Pre-quantum projection
        self.pre_q = nn.Sequential(
            nn.Linear(hidden_size, n_qubits),
            nn.Tanh()
        )

        if quantum_on:
            self.q_module = create_qml_torch_layer(
                n_qubits=n_qubits,
                n_layers=1,
                output_dim=q_out_dim,
                dev_name="default.qubit"
            )
            final_dim = hidden_size + q_out_dim
        else:
            self.q_module = None
            final_dim = hidden_size

        # Category classification head
        self.cat_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, NUM_CAT)
        )

        # Section classification head
        self.sec_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, NUM_SEC)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use pooler output if available
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            hid = out.last_hidden_state
            pooled = (hid * attention_mask.unsqueeze(-1)).sum(1) / (
                attention_mask.sum(1, keepdim=True) + 1e-9
            )

        if self.quantum_on and self.q_module is not None:
            q_in = self.pre_q(pooled) * math.pi
            q_in_cpu = q_in.detach().to(torch.float32).cpu()
            q_emb_cpu = self.q_module(q_in_cpu)
            q_emb = q_emb_cpu.to(pooled.device)
            combined = torch.cat([pooled, q_emb], dim=1)
        else:
            combined = pooled

        logits_cat = self.cat_head(combined)
        logits_sec = self.sec_head(combined)

        return logits_cat, logits_sec