import os
import cv2
import tkinter as tk
from tkinter import Tk, LabelFrame, Button, Entry, Text, Scrollbar, HORIZONTAL, BOTTOM, RIGHT, X, Y, WORD, END
from tkinter import messagebox
from tkinter.filedialog import askopenfilename


class Main_GUI:
    """
    Main GUI for:
    Semantic Context-Aware Automated Front-End Code Generation
    for Mobile Applications using a Vision Language Code Transformer.

    This GUI is intentionally modular. Each processing-stage callback can
    later be connected to the corresponding implementation module.
    """

    def __init__(self, root):
        self.root = root

        # -----------------------------
        # Application state
        # -----------------------------
        self.dataset_loaded = False
        self.selected_image = ""
        self.preprocessing_done = False
        self.preprocessed_image_path = ""
        self.object_detection_done = False
        self.text_attribute_extraction_done = False
        self.grouping_done = False
        self.semantic_relation_done = False
        self.coordinate_alignment_done = False
        self.schema_mapping_done = False
        self.model_trained = False
        self.model_tested = False

        # -----------------------------
        # Fonts / colours
        # -----------------------------
        self.LARGE_FONT = ("Arial", 15, "bold")
        self.text_font1 = ("Constantia", 9)
        self.frame_font = ("Arial", 9, "bold")
        self.frame_process_res_font = ("Arial", 11, "bold")

        self.bg = "#79CDCD"
        self.heading_fg = "#8B2252"
        self.frame_fg = "#8B5A2B"
        self.button_bg = "#1E90FF"
        self.button_fg = "#FFFFFF"

        # -----------------------------
        # Main heading
        # -----------------------------
        heading = tk.Label(
            root,
            text="SEMANTIC CONTEXT-AWARE AUTOMATED FRONT-END CODE GENERATION FOR MOBILE APPLICATIONS",
            fg=self.heading_fg,
            bg=self.bg,
            font=self.LARGE_FONT
        )
        heading.place(x=55, y=5)

        subheading = tk.Label(
            root,
            text="Vision-Language Code Transformer Framework",
            fg="#000080",
            bg=self.bg,
            font=("Arial", 10, "bold")
        )
        subheading.place(x=385, y=32)

        # ==========================================================
        # 1. UI DESIGN DATASET
        # ==========================================================
        self.label_dataset = LabelFrame(
            root, text="UI Design\nDataset",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_dataset.place(x=10, y=55, width=90, height=155)

        self.entry_dataset = Entry(root)
        self.entry_dataset.place(x=20, y=88, width=70, height=20)
        self.entry_dataset.insert(0, r"Dataset\UI")
        self.entry_dataset.configure(state="disabled")

        self.btn_dataset = Button(
            root, text="Read",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=7, height=4,
            command=self.read_ui_dataset
        )
        self.btn_dataset.place(x=20, y=120)

        # ==========================================================
        # 2. IMAGE PREPROCESSING
        # ==========================================================
        self.label_preprocessing = LabelFrame(
            root, text="Image\nPreprocessing",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_preprocessing.place(x=110, y=55, width=100, height=155)

        self.btn_preprocessing = Button(
            root, text="Proceed",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=8, height=5,
            command=self.image_preprocessing
        )
        self.btn_preprocessing.place(x=125, y=100)

        # ==========================================================
        # 3. UI COMPONENT DETECTION - SPP-YOLOv8
        # ==========================================================
        self.label_detection = LabelFrame(
            root, text="UI Component\nDetection",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_detection.place(x=220, y=55, width=120, height=155)

        self.btn_detection_train = Button(
            root, text="Training",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=9,
            command=self.component_detection_training
        )
        self.btn_detection_train.place(x=240, y=90)

        self.btn_detection_test = Button(
            root, text="Testing",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=9,
            command=self.component_detection_testing
        )
        self.btn_detection_test.place(x=240, y=125)

        self.btn_detection = Button(
            root, text="Detection",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=9,
            command=self.ui_component_detection
        )
        self.btn_detection.place(x=240, y=160)

        # ==========================================================
        # 4. TEXT + ATTRIBUTE EXTRACTION
        # ==========================================================
        self.label_extraction = LabelFrame(
            root, text="Text / Attribute\nExtraction",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_extraction.place(x=350, y=55, width=120, height=155)

        self.btn_extraction = Button(
            root, text="Proceed",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=10, height=5,
            command=self.text_attribute_extraction
        )
        self.btn_extraction.place(x=365, y=100)

        # ==========================================================
        # 5. DSFBSCAN COMPONENT GROUPING
        # ==========================================================
        self.label_grouping = LabelFrame(
            root, text="Component\nGrouping",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_grouping.place(x=480, y=55, width=110, height=155)

        self.btn_grouping = Button(
            root, text="DSFBSCAN",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=9, height=5,
            command=self.component_grouping
        )
        self.btn_grouping.place(x=495, y=100)

        # ==========================================================
        # 6. VISUALTABERT SEMANTIC RELATION EXTRACTION
        # ==========================================================
        self.label_semantic = LabelFrame(
            root, text="Semantic\nRelations",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_semantic.place(x=600, y=55, width=110, height=155)

        self.btn_semantic = Button(
            root, text="VisualTABERT",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=10, height=5,
            command=self.semantic_relation_extraction
        )
        self.btn_semantic.place(x=612, y=100)

        # ==========================================================
        # 7. APKT COORDINATE ALIGNMENT
        # ==========================================================
        self.label_alignment = LabelFrame(
            root, text="Coordinate\nAlignment",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_alignment.place(x=720, y=55, width=105, height=155)

        self.btn_alignment = Button(
            root, text="APKT",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=8, height=5,
            command=self.coordinate_alignment
        )
        self.btn_alignment.place(x=735, y=100)

        # ==========================================================
        # 8. SCHEMA MAPPING
        # ==========================================================
        self.label_schema = LabelFrame(
            root, text="Schema\nMapping",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_schema.place(x=835, y=55, width=100, height=155)

        self.btn_schema = Button(
            root, text="Proceed",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=8, height=5,
            command=self.schema_mapping
        )
        self.btn_schema.place(x=850, y=100)

        # ==========================================================
        # 9. CODELAT5+ FRONT-END CODE GENERATION
        # ==========================================================
        self.label_codegen = LabelFrame(
            root, text="Front-End Code\nGeneration",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_codegen.place(x=945, y=55, width=120, height=205)

        self.btn_codegen_train = Button(
            root, text="Training",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=10,
            command=self.code_generation_training
        )
        self.btn_codegen_train.place(x=960, y=90)

        self.btn_codegen_test = Button(
            root, text="Testing",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=10,
            command=self.code_generation_testing
        )
        self.btn_codegen_test.place(x=960, y=130)

        self.btn_codegen = Button(
            root, text="Generate Code",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=10, height=2,
            command=self.generate_frontend_code
        )
        self.btn_codegen.place(x=960, y=175)

        # ==========================================================
        # SELECT SINGLE UI DESIGN IMAGE
        # ==========================================================
        self.label_input = LabelFrame(
            root,
            text="Select UI Design Image for Front-End Code Generation",
            bg=self.bg, fg=self.frame_fg, font=self.frame_font
        )
        self.label_input.place(x=10, y=220, width=920, height=50)

        self.entry_input = Entry(root)
        self.entry_input.place(x=20, y=240, width=820, height=20)

        self.btn_browse = Button(
            root, text="Browse",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            command=self.select_ui_image
        )
        self.btn_browse.place(x=855, y=238)

        # ==========================================================
        # EXTRA CONTROLS
        # ==========================================================
        self.btn_graphs = Button(
            root, text="Generate\nGraphs",
            bg=self.button_bg, fg=self.button_fg,
            font=self.text_font1,
            width=8, height=2,
            command=self.generate_graphs
        )
        self.btn_graphs.place(x=1075, y=75)

        self.btn_clear = Button(
            root, text="Clear",
            width=7,
            command=self.clear
        )
        self.btn_clear.place(x=1080, y=175)

        self.btn_exit = Button(
            root, text="Exit",
            width=7,
            command=self.exit
        )
        self.btn_exit.place(x=1080, y=215)

        # ==========================================================
        # PROCESS / RESULT WINDOWS
        # ==========================================================
        self.process_frame = LabelFrame(
            root, text="Process Window",
            bg=self.bg, fg="#0000FF",
            font=self.frame_process_res_font
        )
        self.process_frame.place(x=10, y=285, width=650, height=330)

        self.result_frame = LabelFrame(
            root, text="Result Window",
            bg=self.bg, fg="#0000FF",
            font=self.frame_process_res_font
        )
        self.result_frame.place(x=675, y=285, width=485, height=330)

        self.process_scroll_y = Scrollbar(root)
        self.process_scroll_y.place(x=635, y=310, height=290)

        self.data_textarea_process = Text(
            root, wrap=WORD,
            yscrollcommand=self.process_scroll_y.set
        )
        self.data_textarea_process.place(x=20, y=310, width=615, height=290)
        self.process_scroll_y.config(command=self.data_textarea_process.yview)
        self.data_textarea_process.configure(state="disabled")

        self.result_scroll_y = Scrollbar(root)
        self.result_scroll_y.place(x=1135, y=310, height=290)

        self.data_textarea_result = Text(
            root, wrap=WORD,
            yscrollcommand=self.result_scroll_y.set
        )
        self.data_textarea_result.place(x=685, y=310, width=450, height=290)
        self.result_scroll_y.config(command=self.data_textarea_result.yview)
        self.data_textarea_result.configure(state="disabled")

        # -----------------------------
        # Create required folders
        # -----------------------------
        self.create_project_folders()

        self.log_process(
            "Semantic Context-Aware Front-End Code Generation GUI initialized.\n"
            "Select a dataset for training/evaluation or browse a UI design image "
            "for single-image code generation."
        )

    # ==============================================================
    # COMMON HELPERS
    # ==============================================================

    def log_process(self, message):
        self.data_textarea_process.configure(state="normal")
        self.data_textarea_process.insert(END, "\n" + str(message))
        self.data_textarea_process.see(END)
        self.data_textarea_process.configure(state="disabled")

    def log_result(self, message):
        self.data_textarea_result.configure(state="normal")
        self.data_textarea_result.insert(END, "\n" + str(message))
        self.data_textarea_result.see(END)
        self.data_textarea_result.configure(state="disabled")

    def create_project_folders(self):
        folders = [
            r"..\Models",
            r"..\Graphs",
            r"..\Output",
            r"..\Output\UI",
            r"..\Output\UI\Preprocessing",
            r"..\Output\UI\Component_Detection",
            r"..\Output\UI\Text_Attribute_Extraction",
            r"..\Output\UI\Component_Grouping",
            r"..\Output\UI\Semantic_Relations",
            r"..\Output\UI\Coordinate_Alignment",
            r"..\Output\UI\Schema_Mapping",
            r"..\Output\Generated_Frontend_Code",
        ]

        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    # ==============================================================
    # STAGE 1 - UI DATASET
    # ==============================================================

    def read_ui_dataset(self):
        dataset_path = r"..\Dataset\UI"

        if not os.path.exists(dataset_path):
            messagebox.showwarning(
                "Dataset",
                f"Dataset folder was not found:\n{dataset_path}"
            )
            self.log_process(f"Dataset folder not found: {dataset_path}")
            return

        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        image_count = 0

        for root_dir, _, files in os.walk(dataset_path):
            image_count += sum(
                1 for f in files if f.lower().endswith(image_extensions)
            )

        self.dataset_loaded = True
        self.log_process("UI Design Dataset")
        self.log_process("=================")
        self.log_process(f"Dataset path: {dataset_path}")
        self.log_process(f"Total UI design images: {image_count}")

        messagebox.showinfo(
            "Info Message",
            "UI design dataset was read successfully."
        )

    # ==============================================================
    # SINGLE IMAGE SELECTION
    # ==============================================================

    def select_ui_image(self):
        file_path = askopenfilename(
            title="Select UI Design Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.selected_image = file_path

        self.entry_input.configure(state="normal")
        self.entry_input.delete(0, END)
        self.entry_input.insert(0, file_path)
        self.entry_input.configure(state="disabled")

        self.log_process("UI design image selected:")
        self.log_process(file_path)

        messagebox.showinfo(
            "Info Message",
            "UI design image was selected successfully."
        )

    # ==============================================================
    # STAGE 2 - IMAGE PREPROCESSING
    # ==============================================================

    def image_preprocessing(self):
        """
        CLAHE-based preprocessing stage.

        Paper notation:
        M_k          -> input UI image
        alpha_i      -> local non-overlapping tiles
        h_epsilon(k) -> normalized/local histogram representation
        C_epsilon_n  -> final contrast-enhanced image
        """

        if not self.selected_image:
            messagebox.showwarning(
                "Input Required",
                "Please select a UI design image first."
            )
            return

        self.log_process("\nImage Preprocessing using CLAHE")
        self.log_process("================================")

        # Read the original UI image M_k.
        image = cv2.imread(self.selected_image)

        if image is None:
            messagebox.showerror(
                "Image Error",
                "Unable to read the selected UI design image."
            )
            self.log_process("Error: selected image could not be read.")
            return

        # Convert the input image into an intensity image.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE divides the image into local tiles alpha_i.
        # clipLimit restricts excessive histogram amplification.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        # Apply local contrast enhancement.
        # OpenCV internally interpolates the mappings of neighboring tiles.
        enhanced = clahe.apply(gray)

        output_dir = r"..\Output\UI\Preprocessing"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.selected_image))[0]
        output_path = os.path.join(
            output_dir,
            base_name + "_CLAHE.png"
        )

        cv2.imwrite(output_path, enhanced)

        # Save stage output for the next stage.
        self.preprocessed_image_path = output_path
        self.preprocessing_done = True

        height, width = gray.shape

        self.log_process(f"Input image (M_k): {self.selected_image}")
        self.log_process(f"Image size: {width} x {height}")
        self.log_process("Local regions (alpha_i): 8 x 8 tiles")
        self.log_process("CLAHE clip limit: 2.0")
        self.log_process("Local histogram redistribution completed.")
        self.log_process("Bilinear interpolation between neighboring tiles completed.")
        self.log_process(
            f"Contrast-enhanced image (C_epsilon_n): {output_path}"
        )

        self.log_result(
            "CLAHE preprocessing completed successfully.\n"
            f"Enhanced image: {output_path}"
        )

        messagebox.showinfo(
            "Info Message",
            "CLAHE preprocessing was completed successfully."
        )

    # ==============================================================
    # STAGE 3 - SPP-YOLOv8 UI COMPONENT DETECTION
    # ==============================================================

    def component_detection_training(self):
        self.log_process("\nUI Component Detection Training")
        self.log_process("================================")
        self.log_process("Proposed model: SPP-YOLOv8")
        self.log_process(
            "Training implementation will be connected to this callback."
        )

    def component_detection_testing(self):
        self.log_process("\nUI Component Detection Testing")
        self.log_process("===============================")
        self.log_process("Proposed model: SPP-YOLOv8")
        self.log_process(
            "Testing/evaluation implementation will be connected here."
        )

    def ui_component_detection(self):
        if not self.selected_image and not self.dataset_loaded:
            messagebox.showwarning(
                "Input Required",
                "Please select a UI design image or load the dataset first."
            )
            return

        self.log_process("\nUI Component Detection")
        self.log_process("======================")
        self.log_process("Model: SPP-YOLOv8")
        self.log_process(
            "Detection implementation will be connected to this callback."
        )

        self.object_detection_done = True

    # ==============================================================
    # STAGE 4 - TEXT AND ATTRIBUTE EXTRACTION
    # ==============================================================

    def text_attribute_extraction(self):
        self.log_process("\nText and Attribute Extraction")
        self.log_process("=============================")
        self.log_process(
            "OCR, component text, bounding-box properties, colour and "
            "visual attributes will be connected here."
        )
        self.text_attribute_extraction_done = True

    # ==============================================================
    # STAGE 5 - DSFBSCAN COMPONENT GROUPING
    # ==============================================================

    def component_grouping(self):
        self.log_process("\nUI Component Grouping")
        self.log_process("=====================")
        self.log_process("Proposed method: DSFBSCAN")
        self.log_process(
            "DSFBSCAN implementation will be connected to this callback."
        )
        self.grouping_done = True

    # ==============================================================
    # STAGE 6 - VISUALTABERT SEMANTIC RELATION EXTRACTION
    # ==============================================================

    def semantic_relation_extraction(self):
        self.log_process("\nSemantic Relationship Extraction")
        self.log_process("================================")
        self.log_process("Proposed model: VisualTABERT")
        self.log_process(
            "Visual-language semantic relation extraction will be connected here."
        )
        self.semantic_relation_done = True

    # ==============================================================
    # STAGE 7 - APKT COORDINATE ALIGNMENT
    # ==============================================================

    def coordinate_alignment(self):
        self.log_process("\nCoordinate Alignment")
        self.log_process("====================")
        self.log_process("Proposed method: APKT")
        self.log_process(
            "Coordinate/layout alignment implementation will be connected here."
        )
        self.coordinate_alignment_done = True

    # ==============================================================
    # STAGE 8 - SCHEMA MAPPING
    # ==============================================================

    def schema_mapping(self):
        self.log_process("\nSchema Mapping")
        self.log_process("==============")
        self.log_process(
            "Detected components, text, groups, semantic relations and "
            "coordinates will be converted into a structured representation here."
        )
        self.schema_mapping_done = True

    # ==============================================================
    # STAGE 9 - CODELAT5+ CODE GENERATION
    # ==============================================================

    def code_generation_training(self):
        self.log_process("\nFront-End Code Generation Training")
        self.log_process("==================================")
        self.log_process("Proposed model: CodeLAT5+")
        self.log_process(
            "Training implementation will be connected to this callback."
        )
        self.model_trained = True

    def code_generation_testing(self):
        self.log_process("\nFront-End Code Generation Testing")
        self.log_process("=================================")
        self.log_process("Proposed model: CodeLAT5+")
        self.log_process(
            "Testing and evaluation metrics will be connected here."
        )
        self.model_tested = True

    def generate_frontend_code(self):
        if not self.selected_image:
            messagebox.showwarning(
                "Input Required",
                "Please select a UI design image first."
            )
            return

        self.log_process("\nFront-End Code Generation")
        self.log_process("=========================")
        self.log_process(f"Input image: {self.selected_image}")
        self.log_process("Generator: CodeLAT5+")

        # Placeholder until the actual CodeLAT5+ implementation is supplied.
        example = (
            "Code generation module is ready to be connected.\n\n"
            "Expected final output:\n"
            "- platform-specific mobile UI source code\n"
            "- component hierarchy\n"
            "- layout properties\n"
            "- component text and attributes\n"
            "- semantic/contextual relationships"
        )

        self.log_result(example)

        messagebox.showinfo(
            "Info Message",
            "Front-end code generation stage executed."
        )

    # ==============================================================
    # RESULT GRAPHS
    # ==============================================================

    def generate_graphs(self):
        self.log_process("\nEvaluation Graph Generation")
        self.log_process("===========================")
        self.log_process(
            "Evaluation/graph code will be connected after the model stages "
            "are implemented."
        )

        messagebox.showinfo(
            "Info Message",
            "Graph-generation callback is ready."
        )

    # ==============================================================
    # WINDOW CONTROLS
    # ==============================================================

    def clear(self):
        self.data_textarea_process.configure(state="normal")
        self.data_textarea_result.configure(state="normal")

        self.data_textarea_process.delete("1.0", END)
        self.data_textarea_result.delete("1.0", END)

        self.data_textarea_process.configure(state="disabled")
        self.data_textarea_result.configure(state="disabled")

        self.entry_input.configure(state="normal")
        self.entry_input.delete(0, END)
        self.entry_input.configure(state="disabled")

        self.selected_image = ""
        self.preprocessing_done = False
        self.object_detection_done = False
        self.text_attribute_extraction_done = False
        self.grouping_done = False
        self.semantic_relation_done = False
        self.coordinate_alignment_done = False
        self.schema_mapping_done = False

        self.log_process("GUI cleared. Select a new UI design image to continue.")

    def exit(self):
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    root.title(
        "Semantic Context-Aware Automated Front-End Code Generation "
        "for Mobile Applications"
    )
    root.geometry("1175x630")
    root.resizable(False, False)
    root.configure(bg="#79CDCD")

    app = Main_GUI(root)
    root.mainloop()
