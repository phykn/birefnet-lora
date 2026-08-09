from pathlib import Path

Pair = tuple[str, str]


def index(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        if stem in result:
            raise ValueError(f"Duplicate stem {stem!r}: {result[stem]} vs {path}")
        result[stem] = path
    return result


def pair_files(image_paths: list[str], mask_paths: list[str]) -> list[Pair]:
    images = index(image_paths)
    masks = index(mask_paths)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            "Image/mask stems do not match: "
            f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]
