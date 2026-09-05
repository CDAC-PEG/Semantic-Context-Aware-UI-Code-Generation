import os
import json
from copy import deepcopy


class Schema_Mapping:
    """
    Structured Intermediate Representation (SM)

    Methodology:
        (Phi_w, G_z, beta_y) -> F_lambda

    Inputs
    ------
    Phi_w:
        Output of APKT coordinate alignment.

    G_z:
        Output of VisualTABERT semantic context extraction.

    beta_y:
        Output of DSFBSCAN similar UI element grouping.

    Output
    ------
    F_lambda:
        Unified structured JSON representation used by the
        front-end code generation stage.
    """

    def __init__(self):
        self.F_lambda = {}

    # ============================================================
    # FILE HELPERS
    # ============================================================

    @staticmethod
    def _load_json(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"JSON file not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    @staticmethod
    def _save_json(data, path):
        folder = os.path.dirname(path)

        if folder:
            os.makedirs(
                folder,
                exist_ok=True
            )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False
            )

    # ============================================================
    # INDEX BUILDERS
    # ============================================================

    @staticmethod
    def _build_group_index(beta_payload):
        """
        Create mapping:
            component_id -> beta_y group
        """

        group_index = {}

        beta_y = beta_payload.get(
            "beta_y",
            {}
        )

        for group_name, members in beta_y.items():

            for member in members:

                component_id = member.get(
                    "component_id"
                )

                if component_id is not None:
                    group_index[
                        str(component_id)
                    ] = group_name

        return group_index

    @staticmethod
    def _build_semantic_group_index(
        semantic_payload
    ):
        """
        Return semantic context indexed by group name.
        """

        return semantic_payload.get(
            "semantic_context",
            {}
        )

    # ============================================================
    # SCHEMA CONSTRUCTION
    # ============================================================

    def build_schema(
        self,
        phi_payload,
        semantic_payload,
        beta_payload
    ):
        """
        Merge Phi_w, G_z, beta_y into F_lambda.
        """

        phi_w = phi_payload.get(
            "Phi_w",
            []
        )

        group_index = (
            self._build_group_index(
                beta_payload
            )
        )

        semantic_index = (
            self._build_semantic_group_index(
                semantic_payload
            )
        )

        semantic_relationships = (
            semantic_payload.get(
                "semantic_relationships",
                []
            )
        )

        structured_components = []

        for element in phi_w:

            component_id = element.get(
                "component_id"
            )

            group_name = group_index.get(
                str(component_id),
                "unassigned"
            )

            semantic_context = {}

            if group_name in semantic_index:
                semantic_context = {
                    "G_z":
                        semantic_index[
                            group_name
                        ].get(
                            "G_z",
                            []
                        ),

                    "top_k_attention_links":
                        semantic_index[
                            group_name
                        ].get(
                            "top_k_attention_links",
                            []
                        )
                }

            structured_component = {
                "component_id":
                    component_id,

                "class_name":
                    element.get(
                        "class_name",
                        ""
                    ),

                "text":
                    element.get(
                        "text",
                        ""
                    ),

                # beta_y
                "group":
                    group_name,

                # Phi_w
                "layout": {
                    "original_coordinates":
                        deepcopy(
                            element.get(
                                "original_coordinates",
                                {}
                            )
                        ),

                    "aligned_coordinates":
                        deepcopy(
                            element.get(
                                "aligned_coordinates",
                                {}
                            )
                        )
                },

                # G_z
                "semantic_context":
                    semantic_context
            }

            structured_components.append(
                structured_component
            )

        # --------------------------------------------------------
        # F_lambda
        # --------------------------------------------------------

        self.F_lambda = {
            "representation":
                "Structured Intermediate Representation",

            "symbol":
                "F_lambda",

            "sources": {
                "coordinate_alignment":
                    "Phi_w",

                "semantic_context":
                    "G_z",

                "similar_ui_groups":
                    "beta_y"
            },

            "components":
                structured_components,

            "semantic_relationships":
                semantic_relationships,

            "group_summary":
                beta_payload.get(
                    "beta_y",
                    {}
                ),

            "metadata": {
                "number_of_components":
                    len(
                        structured_components
                    ),

                "number_of_groups":
                    len(
                        beta_payload.get(
                            "beta_y",
                            {}
                        )
                    ),

                "number_of_semantic_relationships":
                    len(
                        semantic_relationships
                    )
            }
        }

        return self.F_lambda

    # ============================================================
    # COMPLETE FILE-BASED PIPELINE
    # ============================================================

    def create_structured_representation(
        self,
        phi_w_json,
        g_z_json,
        beta_y_json,
        output_json
    ):
        """
        Read three stage outputs and create F_lambda JSON.
        """

        phi_payload = self._load_json(
            phi_w_json
        )

        semantic_payload = self._load_json(
            g_z_json
        )

        beta_payload = self._load_json(
            beta_y_json
        )

        F_lambda = self.build_schema(
            phi_payload,
            semantic_payload,
            beta_payload
        )

        self._save_json(
            F_lambda,
            output_json
        )

        print(
            "\nStructured Intermediate "
            "Representation"
        )

        print(
            "================================"
        )

        print(
            "Total components:",
            F_lambda[
                "metadata"
            ][
                "number_of_components"
            ]
        )

        print(
            "Total groups:",
            F_lambda[
                "metadata"
            ][
                "number_of_groups"
            ]
        )

        print(
            "Semantic relationships:",
            F_lambda[
                "metadata"
            ][
                "number_of_semantic_relationships"
            ]
        )

        print(
            "\nF_lambda saved to:"
        )

        print(
            output_json
        )

        return F_lambda


# ================================================================
# STANDALONE EXAMPLE
# ================================================================

if __name__ == "__main__":

    phi_w_json = (
        r"..\Output\UI\Coordinate_Alignment"
        r"\sample_APKT.json"
    )

    g_z_json = (
        r"..\Output\UI\Semantic_Relations"
        r"\sample_VisualTABERT.json"
    )

    beta_y_json = (
        r"..\Output\UI\Component_Grouping"
        r"\sample_DSFBSCAN.json"
    )

    output_json = (
        r"..\Output\UI\Schema_Mapping"
        r"\sample_F_lambda.json"
    )

    mapper = Schema_Mapping()

    result = (
        mapper.create_structured_representation(
            phi_w_json=phi_w_json,
            g_z_json=g_z_json,
            beta_y_json=beta_y_json,
            output_json=output_json
        )
    )

    print(
        "\nFirst structured component:"
    )

    if result["components"]:
        print(
            json.dumps(
                result["components"][0],
                indent=4
            )
        )
