import json

import numpy as np

from kis.config import VISUAL_CONFIG, milvus_model_properties
from kis.paths import MAX_VIDEO_GROUP, video_group


MILVUS_URI = "http://localhost:19530"
# One collection per visual model; the old SigLIP collection stays untouched.
COLLECTION_NAME = VISUAL_CONFIG["milvus_collection"]
EMBEDDING_DIM = int(VISUAL_CONFIG["dimension"])


class SiglipEncoder:
    """Load the pinned visual model only when encoding is requested."""

    def __init__(self, device="cpu"):
        import torch
        import open_clip
        from huggingface_hub import snapshot_download

        self.device = device
        self.torch = torch
        print(f"[*] Loading SigLIP on {device}...")
        # Same files/loader as tools/embed_siglip2.py, so query and corpus vectors match.
        snapshot = snapshot_download(repo_id=VISUAL_CONFIG["hf_repo"], revision=VISUAL_CONFIG["hf_revision"],
                                     allow_patterns=["open_clip_config.json", "open_clip_model.safetensors",
                                                     "tokenizer*", "special_tokens_map.json"])
        self.model, self.image_preprocess = open_clip.create_model_from_pretrained("local-dir:" + snapshot)
        self.tokenizer = open_clip.get_tokenizer("local-dir:" + snapshot)
        self.model = self.model.to(device)
        self.model.eval()
        print("[*] SigLIP loaded.")

    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with self.torch.no_grad():
            vector = self.model.encode_text(tokens)
        return vector.cpu().numpy().astype("float32")[0]

    def encode_images(self, images):
        if not images:
            return np.empty((0, EMBEDDING_DIM), dtype="float32")
        inputs = self.torch.stack([self.image_preprocess(image) for image in images])
        with self.torch.inference_mode():
            vectors = self.model.encode_image(inputs.to(self.device))
            vectors = self.torch.nn.functional.normalize(vectors.float(), dim=-1)
        return vectors.cpu().numpy().astype("float32")


def validate_collection_model(client, collection_name):
    description = client.describe_collection(
        collection_name=collection_name
    ).get("description", "")
    try:
        actual = json.loads(description)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Milvus collection {collection_name} has no valid model metadata"
        ) from exc

    expected = milvus_model_properties()
    if actual != expected:
        raise RuntimeError(
            f"Milvus collection {collection_name} model metadata does not "
            "match config/models.json"
        )

class SiglipRetriever:
    def __init__(self, video_groups=None, nprobe=10, video_ids=None):
        from pymilvus import MilvusClient

        self.collection_name = COLLECTION_NAME
        self.video_groups = tuple(video_groups or (f"L{n:02d}" for n in range(1, MAX_VIDEO_GROUP + 1)))
        self.video_ids = tuple(sorted(set(video_ids or ())))
        for video_id in self.video_ids:
            if int(video_group(video_id)[1:]) > MAX_VIDEO_GROUP:
                raise ValueError(f"Dataset scope ends at L{MAX_VIDEO_GROUP}: {video_id}")
        for group in self.video_groups:
            if video_group(f"{group}_V001") != group:
                raise ValueError(f"Invalid video group: {group}")
            if int(group[1:]) > MAX_VIDEO_GROUP:
                raise ValueError(f"Dataset scope ends at L{MAX_VIDEO_GROUP}: {group}")
        if not 1 <= nprobe <= 128:
            raise ValueError("nprobe must be between 1 and 128")
        self.nprobe = nprobe
        self.client = MilvusClient(uri=MILVUS_URI)
        validate_collection_model(self.client, self.collection_name)
        print(f"Connected to Milvus: {COLLECTION_NAME}")

    def search(self, query_vector, top_k=100): 
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = np.asarray(query_vector, dtype=np.float32)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("search expects exactly one query vector")

        # Trình chặn lỗi nếu vector khác 768 chiều
        if query_vector.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Query vector has {query_vector.shape[1]} dimensions; "
                f"expected {EMBEDDING_DIM}"
            )
        if not np.isfinite(query_vector).all():
            raise ValueError("Query vector contains NaN or infinity")

        # Chuẩn hóa L2 (Bắt buộc cho search Inner Product - IP)
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Query vector must not be zero")
        query_vector = query_vector / norms

        scope = {}
        if self.video_ids:
            scope["filter"] = "video_id in " + json.dumps(self.video_ids)
        elif self.video_groups:
            scope["filter"] = " or ".join(f'video_id like "{group}_%"' for group in self.video_groups)
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vector.tolist(),
            anns_field="embedding",
            search_params={"metric_type": "IP", "params": {"nprobe": self.nprobe}},
            limit=top_k,
            output_fields=["video_id", "frame_id"],
            **scope,
        )

        return [
            {
                "video_id": result["entity"]["video_id"],
                "frame_id": result["entity"]["frame_id"],
                "score": float(result["distance"])
            }
            for result in results[0]
        ]
