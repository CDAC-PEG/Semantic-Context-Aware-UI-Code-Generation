import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from Proposed_DSFBSCAN import Proposed_DSFBSCAN

from existing_clustering_methods import (
    Existing_DBSCAN,
    Existing_KMeans,
    Existing_FFC,
    Existing_FCM
)


# ================================================================
# LOAD STAGE-2 COMPONENTS
# ================================================================

def load_components(stage2_json_path):
    if not os.path.exists(
        stage2_json_path
    ):
        raise FileNotFoundError(
            f"Stage-2 JSON not found: "
            f"{stage2_json_path}"
        )

    with open(
        stage2_json_path,
        "r",
        encoding="utf-8"
    ) as file:
        payload = json.load(
            file
        )

    return payload.get(
        "components",
        []
    )


# ================================================================
# COMMON FEATURE MATRIX
# ================================================================

def build_common_feature_matrix(
    components
):
    """
    Use the exact same A_h + chi_v feature construction
    as Proposed_DSFBSCAN for every clustering method.
    """

    feature_builder = (
        Proposed_DSFBSCAN()
    )

    X = feature_builder.build_feature_matrix(
        components
    )

    if X.size == 0:
        raise ValueError(
            "No UI component features were found."
        )

    scaler = StandardScaler()

    return scaler.fit_transform(
        X
    )


# ================================================================
# SAFE SILHOUETTE
# ================================================================

def safe_silhouette_score(
    X,
    labels
):
    labels = np.asarray(
        labels
    )

    # DBSCAN marks noise as -1.
    # For a fair cluster-quality measurement,
    # exclude noise from silhouette computation.
    valid = labels != -1

    X_valid = X[
        valid
    ]

    labels_valid = labels[
        valid
    ]

    unique_labels = np.unique(
        labels_valid
    )

    if (
        len(X_valid) < 2
        or
        len(unique_labels) < 2
        or
        len(unique_labels)
        >= len(X_valid)
    ):
        return np.nan

    return float(
        silhouette_score(
            X_valid,
            labels_valid,
            metric="euclidean"
        )
    )


# ================================================================
# PROPOSED DSFBSCAN
# ================================================================

def evaluate_proposed_dsfbscan(
    components,
    X_scaled
):
    model = Proposed_DSFBSCAN(
        survival_quantile=0.70,
        min_minpts=2,
        max_minpts=8
    )

    start = time.perf_counter()

    result = model.cluster_components(
        components
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    labels = np.asarray(
        result["labels"],
        dtype=int
    )

    silhouette = (
        safe_silhouette_score(
            X_scaled,
            labels
        )
    )

    return {
        "Method":
            "DSFBSCAN",

        "Silhouette Score":
            silhouette,

        "Grouping Time (ms)":
            elapsed_ms,

        "Number of Clusters":
            len(
                set(
                    labels.tolist()
                )
                - {-1}
            ),

        "Noise Points":
            int(
                np.sum(
                    labels == -1
                )
            ),

        "zeta":
            result.get(
                "zeta",
                np.nan
            ),

        "P":
            result.get(
                "P",
                np.nan
            )
    }


# ================================================================
# EXISTING METHODS
# ================================================================

def evaluate_existing_method(
    method,
    X_scaled
):
    labels, elapsed_ms = (
        method.cluster(
            X_scaled
        )
    )

    labels = np.asarray(
        labels,
        dtype=int
    )

    silhouette = (
        safe_silhouette_score(
            X_scaled,
            labels
        )
    )

    return {
        "Method":
            method.model_name,

        "Silhouette Score":
            silhouette,

        "Grouping Time (ms)":
            elapsed_ms,

        "Number of Clusters":
            len(
                set(
                    labels.tolist()
                )
                - {-1}
            ),

        "Noise Points":
            int(
                np.sum(
                    labels == -1
                )
            ),

        "zeta":
            np.nan,

        "P":
            np.nan
    }


# ================================================================
# FULL COMPARATIVE ANALYSIS
# ================================================================

def comparative_analysis(
    stage2_json_path,
    output_json=None,
    output_csv=None,
    n_clusters=3,
    dbscan_eps=0.5,
    dbscan_min_samples=5
):
    components = load_components(
        stage2_json_path
    )

    X_scaled = (
        build_common_feature_matrix(
            components
        )
    )

    results = []

    # ------------------------------------------------------------
    # Proposed DSFBSCAN
    # ------------------------------------------------------------

    results.append(
        evaluate_proposed_dsfbscan(
            components,
            X_scaled
        )
    )

    # ------------------------------------------------------------
    # Existing DBSCAN
    # ------------------------------------------------------------

    dbscan = Existing_DBSCAN(
        eps=dbscan_eps,
        min_samples=
            dbscan_min_samples
    )

    results.append(
        evaluate_existing_method(
            dbscan,
            X_scaled
        )
    )

    # ------------------------------------------------------------
    # Existing K-Means
    # ------------------------------------------------------------

    kmeans = Existing_KMeans(
        n_clusters=n_clusters
    )

    results.append(
        evaluate_existing_method(
            kmeans,
            X_scaled
        )
    )

    # ------------------------------------------------------------
    # Farthest First Clustering
    # ------------------------------------------------------------

    ffc = Existing_FFC(
        n_clusters=n_clusters
    )

    results.append(
        evaluate_existing_method(
            ffc,
            X_scaled
        )
    )

    # ------------------------------------------------------------
    # Fuzzy C-Means
    # ------------------------------------------------------------

    fcm = Existing_FCM(
        n_clusters=n_clusters,
        m=2.0,
        max_iter=150,
        tol=1e-5
    )

    results.append(
        evaluate_existing_method(
            fcm,
            X_scaled
        )
    )

    df = pd.DataFrame(
        results
    )

    # Format only for console display.
    display_df = df.copy()

    display_df[
        "Silhouette Score"
    ] = display_df[
        "Silhouette Score"
    ].apply(
        lambda x:
            "N/A"
            if pd.isna(x)
            else f"{x:.4f}"
    )

    display_df[
        "Grouping Time (ms)"
    ] = display_df[
        "Grouping Time (ms)"
    ].apply(
        lambda x:
            f"{x:.2f}"
    )

    print(
        "\nClustering Comparative Analysis"
    )

    print(
        "==============================="
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # EXISTING-METHOD AVERAGES
    # ------------------------------------------------------------

    existing_df = df[
        df["Method"]
        != "DSFBSCAN"
    ]

    avg_silhouette = (
        existing_df[
            "Silhouette Score"
        ].mean(
            skipna=True
        )
    )

    avg_time = (
        existing_df[
            "Grouping Time (ms)"
        ].mean()
    )

    print(
        "\nAverage existing-method "
        "Silhouette Score:",
        round(
            float(
                avg_silhouette
            ),
            4
        )
    )

    print(
        "Average existing-method "
        "Grouping Time (ms):",
        round(
            float(
                avg_time
            ),
            2
        )
    )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    if output_json:

        folder = os.path.dirname(
            output_json
        )

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        json_records = (
            df.where(
                pd.notnull(df),
                None
            )
            .to_dict(
                orient="records"
            )
        )

        with open(
            output_json,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                json_records,
                file,
                indent=4
            )

    if output_csv:

        folder = os.path.dirname(
            output_csv
        )

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        df.to_csv(
            output_csv,
            index=False
        )

    return df


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    comparative_analysis(

        stage2_json_path=
            r"..\Output\UI\Stage2"
            r"\sample_stage2.json",

        output_json=
            r"..\Output\UI\Component_Grouping"
            r"\clustering_comparison.json",

        output_csv=
            r"..\Output\UI\Component_Grouping"
            r"\clustering_comparison.csv",

        n_clusters=3,

        dbscan_eps=0.5,

        dbscan_min_samples=5
    )
