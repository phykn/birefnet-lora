import numpy as np


def random_flip(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random element of the D4 dihedral group to (image, mask).

    The eight group elements are indexed by (rot_k in 0..3, hflip in 0..1).
    """
    rot_k = int(np.random.randint(4))
    do_hflip = bool(np.random.randint(2))
    return _apply(image, rot_k, do_hflip), _apply(mask, rot_k, do_hflip)


def _apply(x: np.ndarray, rot_k: int, do_hflip: bool) -> np.ndarray:
    y = np.rot90(x, k=rot_k, axes=(0, 1))
    if do_hflip:
        y = np.flip(y, axis=1)
    return np.ascontiguousarray(y)
