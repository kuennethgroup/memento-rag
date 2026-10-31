# Every image is named with its registry, so Podman without a registry search list (a fresh account) can pull it; only
# the images scripts/build_images.sh builds locally keep a short name.
import re

from memento.config.paths import PROJECT_ROOT

LOCAL = {"mineru", "memento-fileserver", "qwen3-vl-reasoning"}


def _qualified(ref: str) -> bool:
    first = ref.split("/")[0]
    return "/" in ref and ("." in first or first == "localhost")


def test_dockerfiles_name_their_registry():
    for f in sorted((PROJECT_ROOT / "docker").glob("Dockerfile*")):
        for ref in re.findall(r"^FROM\s+(\S+)", f.read_text(), re.M):
            assert _qualified(ref), f"{f.name}: {ref}"


def test_compose_images_name_their_registry():
    for ref in re.findall(r"^\s*image:\s*(\S+)", (PROJECT_ROOT / "docker-compose.yml").read_text(), re.M):
        if ref.split(":")[0] not in LOCAL:
            assert _qualified(ref), ref


def test_the_reranker_starts_from_the_published_weights():
    # vLLM 0.30 reads tie_word_embeddings from the text config, which the published Qwen3-VL-Reranker-8B config lacks
    # ("'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'" on a fresh download)
    import json
    import yaml
    command = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())["services"]["reranker"]["command"]
    overrides = json.loads(command[command.index("--hf_overrides") + 1])
    assert overrides["text_config"] == {"tie_word_embeddings": False}
    assert overrides["architectures"] == ["Qwen3VLForSequenceClassification"]
