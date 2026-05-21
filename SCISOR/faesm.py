import os

from torch import nn

from .esm import FAEsmForMaskedLM


DEFAULT_HF_CACHE = "/public/home/zhangyangroup/chengshiz/.cache/huggingface/hub"


def resolve_local_hf_snapshot(model_name, cache_dir=DEFAULT_HF_CACHE):
    cache_dir = os.environ.get("BIODEL_HF_CACHE", cache_dir)
    if os.path.isdir(model_name):
        return model_name
    if "/" not in model_name:
        return model_name
    namespace, repo = model_name.split("/", 1)
    repo_dir = os.path.join(cache_dir, "models--{}--{}".format(namespace, repo))
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    refs_main = os.path.join(repo_dir, "refs", "main")
    if not os.path.isdir(snapshots_dir):
        return model_name
    snapshot = None
    if os.path.exists(refs_main):
        with open(refs_main) as handle:
            ref = handle.read().strip()
        candidate = os.path.join(snapshots_dir, ref)
        if os.path.isdir(candidate):
            snapshot = candidate
    if snapshot is None:
        candidates = [
            os.path.join(snapshots_dir, name)
            for name in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        if candidates:
            snapshot = sorted(candidates)[-1]
    if snapshot and os.path.exists(os.path.join(snapshot, "config.json")):
        return snapshot
    return model_name


class FAESM_Base(nn.Module):
    def __init__(self, hf_model_name="esm2_t6_8M_UR50D", **kwargs):
        super().__init__()
        print(f"Using FAESM model {hf_model_name}")
        conditioning_dim = kwargs.get("d_embedding", 128)
        pretrained = kwargs.get("pretrained", True)
        model_name_or_path = resolve_local_hf_snapshot(f"facebook/{hf_model_name}")
        print(f"Loading FAESM backbone from {model_name_or_path}")

        self.faesm = FAEsmForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            use_fa=True,
            conditioning_dim=conditioning_dim,
            load_pretrained_weights=pretrained,
        )
        self.embed_dim = (
            self.faesm.esm.embeddings.word_embeddings.embedding_dim
        )  # 320 for smallest ESM, 480 for 35M
        self.proj = nn.Linear(self.embed_dim, 1)

    def forward(self, x, t, input_mask=None, S=None):
        cond = t if S is None else S
        embeddings = self.faesm(
            input_ids=x, attention_mask=input_mask, conditioning=cond
        )["last_hidden_state"]
        preds = self.proj(embeddings).squeeze()
        return preds
