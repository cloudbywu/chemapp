"""Retrieval-augmented molecular structure prediction from NMR spectra.

Builds a database of spectrum embeddings from training molecules,
then for a query spectrum finds the most similar known molecules.
"""
import torch
import json
import time
from pathlib import Path

from rdkit import Chem

from app.ml.models.nmr_encoder import NUCLEUS_VOCAB, NMREncoder

# Legacy prototype chain (2026-08 ML review): retrieval over rule-synthesised
# demo embeddings.  The peak-token nucleus indices below MUST come from
# nmr_encoder.NUCLEUS_VOCAB ("13C"/"1H"); the module previously carried its
# own H/C/N/O... vocab with the opposite index order, which was dead code.
CKPT_DIR = Path(__file__).parent / "pretrained"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "graphdiff_cache"
DB_PATH = Path(__file__).parent.parent.parent / "data" / "retrieval_db.pt"


def canonical_smiles_from_shard(data) -> str:
    """Return a canonical SMILES from a shard record, failing closed.

    Atom-type strings are not SMILES; the shard must carry a real structure
    field (``smiles``, ``canonical_smiles`` or ``inchikey``).
    """
    raw_smiles = (
        getattr(data, "smiles", None)
        or getattr(data, "canonical_smiles", None)
        or getattr(data, "inchikey", None)
    )
    if not raw_smiles:
        raise NotImplementedError(
            "retrieval shard lacks smiles/canonical_smiles/inchikey; "
            "refusing to build a database with fabricated atom-type strings"
        )
    molecule = Chem.MolFromSmiles(str(raw_smiles))
    if molecule is None and str(raw_smiles).startswith("InChI="):
        molecule = Chem.MolFromInchi(str(raw_smiles))
    if molecule is None:
        raise ValueError(f"unparseable structure in retrieval shard: {raw_smiles}")
    return Chem.MolToSmiles(molecule)


class RetrievalDB:
    """Database of pre-computed spectrum embeddings for similarity search."""

    def __init__(self, embeddings: torch.Tensor, smiles_list: list[str], shift_data: list[dict]):
        self.embeddings = embeddings  # (M, 256)
        self.smiles_list = smiles_list
        self.shift_data = shift_data
        # Normalize for cosine similarity
        self.embeddings_norm = torch.nn.functional.normalize(embeddings, dim=-1)

    @classmethod
    def build(cls, encoder: NMREncoder, device: torch.device, max_molecules: int = 10000) -> "RetrievalDB":
        """Build database by encoding all training molecules."""
        print(f"Building retrieval database (max {max_molecules} molecules)...")
        with open(CACHE_DIR / "manifest.json") as f:
            manifest = json.load(f)

        embeddings_list = []
        smiles_list = []
        shift_data_list = []
        t0 = time.time()
        max_peaks = 48

        for info in manifest["shards"]:
            shard = torch.load(CACHE_DIR / info["filename"], weights_only=False)
            for data in shard:
                if len(embeddings_list) >= max_molecules:
                    break
                # Only use molecules with 13C shifts
                if not hasattr(data, "node_shift_13c_mask") or not data.node_shift_13c_mask.any().item():
                    continue
                N = data.x.shape[0]

                # Build peak list
                s13, s1h = [], []
                if hasattr(data, "node_shift_13c_mask"):
                    for i in range(N):
                        if data.node_shift_13c_mask[i]:
                            s13.append(data.node_shift_13c[i].item())
                if hasattr(data, "node_shift_1h_mask"):
                    for i in range(N):
                        if data.node_shift_1h_mask[i]:
                            s1h.append(data.node_shift_1h[i].item())

                st = torch.zeros(1, max_peaks)
                nt = torch.zeros(1, max_peaks, dtype=torch.long)
                mt = torch.zeros(1, max_peaks, dtype=torch.long)
                it = torch.ones(1, max_peaks)
                mkt = torch.zeros(1, max_peaks, dtype=torch.bool)
                pi = 0
                for s in s13:
                    if pi >= max_peaks:
                        break
                    st[0, pi] = s
                    nt[0, pi] = NUCLEUS_VOCAB["13C"]
                    mt[0, pi] = 0
                    it[0, pi] = 1.0
                    mkt[0, pi] = True
                    pi += 1
                for s in s1h:
                    if pi >= max_peaks:
                        break
                    st[0, pi] = s
                    nt[0, pi] = NUCLEUS_VOCAB["1H"]
                    mt[0, pi] = 0
                    it[0, pi] = 1.0
                    mkt[0, pi] = True
                    pi += 1

                # Encode
                with torch.no_grad():
                    z_spec, _ = encoder(st.to(device), nt.to(device), mt.to(device), it.to(device), mkt.to(device))
                embeddings_list.append(z_spec[0].cpu())

                # SMILES must come from the shard, never fabricated from atom
                # types (atom-type strings are not valid SMILES).
                smiles_list.append(canonical_smiles_from_shard(data))
                shift_data_list.append({"n_peaks_13c": len(s13), "n_peaks_1h": len(s1h), "shifts_13c": s13[:5]})

                if len(embeddings_list) % 2000 == 0:
                    print(f"  {len(embeddings_list)}/{max_molecules} ({time.time() - t0:.0f}s)")

            if len(embeddings_list) >= max_molecules:
                break

        print(f"Database: {len(embeddings_list)} molecules in {time.time() - t0:.0f}s")
        return cls(torch.stack(embeddings_list), smiles_list, shift_data_list)

    def search(self, z_query: torch.Tensor, top_k: int = 10) -> list[dict]:
        """Find top-K most similar molecules by cosine similarity."""
        z_norm = torch.nn.functional.normalize(z_query.unsqueeze(0) if z_query.dim() == 1 else z_query, dim=-1)
        emb = self.embeddings_norm.to(z_query.device)
        scores = (z_norm @ emb.T).squeeze(0)
        top_scores, top_indices = scores.topk(min(top_k, len(self.smiles_list)))
        results = []
        for i, (idx, score) in enumerate(zip(top_indices.tolist(), top_scores.tolist())):
            result = {
                "rank": i + 1,
                "smiles": self.smiles_list[idx],
                "similarity": round(score, 4),
            }
            if idx < len(self.shift_data):
                result["n_peaks_13c"] = self.shift_data[idx].get("n_peaks_13c", 0) if self.shift_data[idx] else 0
            results.append(result)
        return results

    def save(self, path: Path | None = None):
        path = path or DB_PATH
        torch.save({
            "embeddings": self.embeddings,
            "smiles_list": self.smiles_list,
            "shift_data": self.shift_data,
        }, path)
        print(f"Saved database to {path}")

    @classmethod
    def load(cls, path: Path | None = None, device: torch.device | None = None) -> "RetrievalDB":
        path = path or DB_PATH
        data = torch.load(path, map_location=device or "cpu", weights_only=True)
        return cls(data["embeddings"], data["smiles_list"], data["shift_data"])


def predict_with_retrieval(
    encoder: NMREncoder,
    db: RetrievalDB,
    peaks_13c: list[float],
    peaks_1h: list[float] | None = None,
    top_k: int = 10,
    device: torch.device | None = None,
) -> list[dict]:
    """Predict molecular candidates using spectrum similarity retrieval."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_peaks = 48

    # Build peak tensors
    st = torch.zeros(1, max_peaks, device=device)
    nt = torch.zeros(1, max_peaks, dtype=torch.long, device=device)
    mt = torch.zeros(1, max_peaks, dtype=torch.long, device=device)
    it = torch.ones(1, max_peaks, device=device)
    mkt = torch.zeros(1, max_peaks, dtype=torch.bool, device=device)
    pi = 0
    for s in peaks_13c:
        if pi >= max_peaks:
            break
        st[0, pi] = s
        nt[0, pi] = NUCLEUS_VOCAB["13C"]
        mt[0, pi] = 0
        it[0, pi] = 1.0
        mkt[0, pi] = True
        pi += 1
    if peaks_1h:
        for s in peaks_1h:
            if pi >= max_peaks:
                break
            st[0, pi] = s
            nt[0, pi] = NUCLEUS_VOCAB["1H"]
            mt[0, pi] = 0
            it[0, pi] = 1.0
            mkt[0, pi] = True
            pi += 1

    # Encode query
    encoder.eval()
    with torch.no_grad():
        z_q, _ = encoder(st, nt, mt, it, mkt)

    # Search
    return db.search(z_q[0], top_k=top_k)


if __name__ == "__main__":
    # Build and test the retrieval database
    device = torch.device("cuda")
    print(f"Device: {device}")

    # Load encoder
    encoder = NMREncoder(d_model=256, n_heads=8, n_layers=4).to(device)
    ckpt = torch.load(CKPT_DIR / "stage2_final.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["model_state"])
    encoder.eval()
    print(f"Encoder: epoch {ckpt.get('epoch','?')}")

    # Build database
    if DB_PATH.exists():
        print(f"Loading existing database from {DB_PATH}")
        db = RetrievalDB.load(DB_PATH)
    else:
        db = RetrievalDB.build(encoder, device, max_molecules=5000)
        db.save()

    print(f"\nDatabase: {len(db.smiles_list)} molecules")

    # Test retrieval with a random query
    print("\n=== Test retrieval ===")
    test_peaks_13c = [128.5, 128.5, 128.5, 128.5, 128.5, 128.5]  # benzene-like
    results = predict_with_retrieval(encoder, db, test_peaks_13c, top_k=5)
    for r in results:
        print(f"  #{r['rank']}: sim={r['similarity']:.3f}, 13C={r['n_peaks_13c']}peaks, composition={r['smiles'][:40]}")
