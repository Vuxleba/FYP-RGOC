import argparse
import sys
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

from utils import clustering, load_graph_data, setup_seed


def get_facebook_pca_dim(dataset_name: str) -> int:
    """Mirror the PCA routing used by the existing FCM runner."""
    ego_id = dataset_name.split("_")[1]
    mapping = {
        "0": 128,
        "107": 256,
        "348": 64,
        "414": 32,
        "686": 32,
        "698": 32,
        "1684": 128,
        "1912": 256,
        "3437": 128,
        "3980": 32,
    }
    return mapping.get(ego_id, -1)


def to_community_sets(binary_membership: torch.Tensor, include_empty: bool = True) -> List[Set[int]]:
    """Convert an (N, K) binary membership matrix to list-of-sets communities."""
    membership = binary_membership.detach().cpu().numpy()
    communities: List[Set[int]] = []
    for c in range(membership.shape[1]):
        nodes = set(np.where(membership[:, c] > 0.5)[0].tolist())
        if include_empty or len(nodes) > 0:
            communities.append(nodes)
    return communities


def to_cdlib_node_clustering(communities: Iterable[Set[int]], method_name: str):
    """Create a CDlib NodeClustering object for overlapping partitions."""
    try:
        from cdlib import NodeClustering
    except ImportError as exc:
        raise ImportError(
            "CDlib is not installed. Install with: pip install cdlib"
        ) from exc

    cleaned = [sorted(list(c)) for c in communities if len(c) > 0]
    return NodeClustering(communities=cleaned, graph=None, method_name=method_name, overlap=True)


def resolve_cdlib_metric_functions() -> Tuple[Callable, str, Callable, str]:
    """Resolve ONMI and F1 metric functions across CDlib versions."""
    try:
        from cdlib import evaluation
    except ImportError as exc:
        raise ImportError(
            "CDlib is not installed. Install with: pip install cdlib"
        ) from exc

    onmi_candidates = [
        "overlapping_normalized_mutual_information_MGH",
        "overlapping_normalized_mutual_information_LFK",
        "overlapping_normalized_mutual_information",
    ]
    f1_candidates = ["f1", "nf1"]

    onmi_func = None
    onmi_name = ""
    for name in onmi_candidates:
        if hasattr(evaluation, name):
            onmi_func = getattr(evaluation, name)
            onmi_name = name
            break

    f1_func = None
    f1_name = ""
    for name in f1_candidates:
        if hasattr(evaluation, name):
            f1_func = getattr(evaluation, name)
            f1_name = name
            break

    if onmi_func is None:
        raise RuntimeError(
            "Could not find an overlapping NMI function in cdlib.evaluation. "
            "Checked: " + ", ".join(onmi_candidates)
        )
    if f1_func is None:
        raise RuntimeError(
            "Could not find an F1 function in cdlib.evaluation. "
            "Checked: " + ", ".join(f1_candidates)
        )

    return onmi_func, onmi_name, f1_func, f1_name


def metric_result_to_float(result) -> float:
    """Extract a numeric score from CDlib metric outputs."""
    if hasattr(result, "score"):
        value = result.score
    else:
        value = result

    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) == 0:
            return float("nan")
        value = value[0]

    return float(value)


def maybe_percent(value: float) -> float:
    """Convert to percent if a metric appears to be in [0, 1]."""
    if np.isnan(value):
        return value
    if 0.0 <= value <= 1.0:
        return 100.0 * value
    return value


def run_single_seed(
    seed: int,
    dataset_name: str,
    device: str,
    onmi_func: Callable,
    f1_func: Callable,
) -> Dict[str, float]:
    setup_seed(seed)

    features, true_communities, _ = load_graph_data(dataset_name, show_details=False)
    pca_dim = get_facebook_pca_dim(dataset_name)

    if pca_dim != -1 and features.shape[1] > pca_dim:
        features = PCA(n_components=pca_dim).fit_transform(features)

    features_tensor = torch.from_numpy(features).float().to(device)
    cluster_num = len(true_communities)

    custom_nmi, custom_f1, _, _, pred_binary_membership, _, _ = clustering(
        features_tensor,
        true_communities,
        cluster_num,
        device=device,
    )

    pred_communities = to_community_sets(pred_binary_membership, include_empty=True)
    pred_partition = to_cdlib_node_clustering(pred_communities, method_name="fcm")
    true_partition = to_cdlib_node_clustering(true_communities, method_name="ground_truth")

    cdlib_onmi = maybe_percent(metric_result_to_float(onmi_func(pred_partition, true_partition)))

    # F1 in CDlib is directional in some versions; evaluate both directions and average.
    f1_forward = maybe_percent(metric_result_to_float(f1_func(pred_partition, true_partition)))
    f1_reverse = maybe_percent(metric_result_to_float(f1_func(true_partition, pred_partition)))
    cdlib_f1 = 0.5 * (f1_forward + f1_reverse)

    non_empty_pred = sum(1 for c in pred_communities if len(c) > 0)
    return {
        "seed": float(seed),
        "custom_onmi": float(custom_nmi),
        "cdlib_onmi": float(cdlib_onmi),
        "custom_f1": float(custom_f1),
        "cdlib_f1": float(cdlib_f1),
        "pred_k": float(len(pred_communities)),
        "pred_k_non_empty": float(non_empty_pred),
        "true_k": float(len(true_communities)),
    }


def summarize(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def parse_seeds(seeds_csv: str) -> List[int]:
    values = [s.strip() for s in seeds_csv.split(",") if s.strip()]
    if not values:
        raise ValueError("No seeds provided. Use --seeds like: 0,1,2,3,4")
    return [int(v) for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare custom overlap ONMI/F1 vs CDlib on facebook_0 using FCM."
    )
    parser.add_argument("--dataset", type=str, default="facebook_3980")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g., cpu or cuda:0",
    )
    args = parser.parse_args()

    if args.dataset != "facebook_0":
        print(f"Warning: this script is intended for facebook_0, got {args.dataset}")

    seeds = parse_seeds(args.seeds)
    onmi_func, onmi_name, f1_func, f1_name = resolve_cdlib_metric_functions()

    print("=" * 96)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Seeds: {seeds}")
    print(f"CDlib ONMI function: {onmi_name}")
    print(f"CDlib F1 function: {f1_name} (reported as bidirectional average)")
    print("Thresholding rule: normalized membership > 0.5")
    print("=" * 96)

    rows = []
    for seed in seeds:
        row = run_single_seed(
            seed=seed,
            dataset_name=args.dataset,
            device=args.device,
            onmi_func=onmi_func,
            f1_func=f1_func,
        )
        row["abs_diff_onmi"] = abs(row["custom_onmi"] - row["cdlib_onmi"])
        row["abs_diff_f1"] = abs(row["custom_f1"] - row["cdlib_f1"])
        rows.append(row)

        print(
            "Seed {seed:>3} | "
            "custom_onmi={custom_onmi:>7.3f} cdlib_onmi={cdlib_onmi:>7.3f} diff={abs_diff_onmi:>7.3f} | "
            "custom_f1={custom_f1:>7.3f} cdlib_f1={cdlib_f1:>7.3f} diff={abs_diff_f1:>7.3f} | "
            "true_k={true_k:.0f} pred_k_non_empty={pred_k_non_empty:.0f}"
            .format(**row)
        )

    custom_onmi_mean, custom_onmi_std = summarize([r["custom_onmi"] for r in rows])
    cdlib_onmi_mean, cdlib_onmi_std = summarize([r["cdlib_onmi"] for r in rows])
    diff_onmi_mean, diff_onmi_std = summarize([r["abs_diff_onmi"] for r in rows])

    custom_f1_mean, custom_f1_std = summarize([r["custom_f1"] for r in rows])
    cdlib_f1_mean, cdlib_f1_std = summarize([r["cdlib_f1"] for r in rows])
    diff_f1_mean, diff_f1_std = summarize([r["abs_diff_f1"] for r in rows])

    print("-" * 96)
    print(
        "ONMI  custom mean/std: {0:.3f}/{1:.3f} | cdlib mean/std: {2:.3f}/{3:.3f} | abs diff mean/std: {4:.3f}/{5:.3f}".format(
            custom_onmi_mean,
            custom_onmi_std,
            cdlib_onmi_mean,
            cdlib_onmi_std,
            diff_onmi_mean,
            diff_onmi_std,
        )
    )
    print(
        "F1    custom mean/std: {0:.3f}/{1:.3f} | cdlib mean/std: {2:.3f}/{3:.3f} | abs diff mean/std: {4:.3f}/{5:.3f}".format(
            custom_f1_mean,
            custom_f1_std,
            cdlib_f1_mean,
            cdlib_f1_std,
            diff_f1_mean,
            diff_f1_std,
        )
    )
    print("=" * 96)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
