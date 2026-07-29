import os

from silab_triple_judge_common import model_slug, project_root, run_validation


GENERATOR_MODEL_NAME = os.getenv("SILAB_GENERATOR_MODEL", "gpt-oss:120b")
GENERATOR_SLUG = model_slug(GENERATOR_MODEL_NAME)

PROJECT_ROOT = project_root()
INPUT_GOLD = os.getenv(
    "SILAB_MPTP_INPUT_GOLD",
    PROJECT_ROOT / "silab" / "model" / f"canonical_gold_qas_{GENERATOR_SLUG}_MPTP.json",
)
VERSE_FILE = os.getenv(
    "SILAB_MPTP_VERSE_FILE",
    PROJECT_ROOT / "canonical_verses_with_theme.json",
)


if __name__ == "__main__":
    run_validation(
        dataset_type="MPTP",
        input_gold=INPUT_GOLD,
        generator_model_name=GENERATOR_MODEL_NAME,
        verse_file=VERSE_FILE,
    )
