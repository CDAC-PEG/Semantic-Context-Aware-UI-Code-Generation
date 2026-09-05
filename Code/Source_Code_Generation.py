import cv2
import pytesseract
import json
import os
import re

# === CONFIGURATION ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# =====================

def extract_classes_from_image(image_path):
    """Detect UML class boxes and extract OCR text."""
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    class_data = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 120 and h > 80:  # likely a UML class/interface/enum box
            roi = img[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, config="--psm 6")
            parsed = parse_uml_text(text)
            if parsed:
                class_data.append(parsed)

    return class_data

def parse_uml_text(text_block):
    """Extract UML info (class/interface/enum, attributes, methods, relations)."""
    lines = [line.strip() for line in text_block.split("\n") if line.strip()]
    if not lines:
        return None

    first_line = lines[0]
    class_type = "class"

    # Identify type
    if "interface" in first_line.lower():
        class_type = "interface"
        class_name = re.sub(r'(?i)interface', '', first_line).strip()
    elif "enum" in first_line.lower():
        class_type = "enum"
        class_name = re.sub(r'(?i)enum', '', first_line).strip()
    else:
        class_name = first_line.strip()

    attributes, methods = [], []
    extends, implements = None, []

    divider_found = False
    for line in lines[1:]:
        line = line.replace("–", "-")  # fix OCR dash issue

        # Detect inheritance and interface implementation
        if "extends" in line:
            match = re.search(r'extends\s+(\w+)', line)
            if match:
                extends = match.group(1)

        if "implements" in line:
            match = re.search(r'implements\s+([\w, ]+)', line)
            if match:
                implements = [i.strip() for i in match.group(1).split(",")]

        # Split into attributes and methods
        if "(" in line and ")" in line:
            divider_found = True

        if not divider_found:
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    name, typ = parts
                    attributes.append({"name": name.strip(), "type": typ.strip()})
        else:
            if "(" in line and ")" in line:
                method_name = line.split("(")[0].strip()
                methods.append({
                    "name": method_name,
                    "return_type": "void",
                    "parameters": []
                })

    return {
        "name": class_name,
        "type": class_type,
        "extends": extends,
        "implements": implements,
        "attributes": attributes,
        "methods": methods
    }

def save_as_json(class_data, output_path="uml_classes.json"):
    uml_json = {"classes": class_data}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(uml_json, f, indent=4)

def generate_java_class(class_info, output_dir):
    """Generate Java source file from UML info."""
    os.makedirs(output_dir, exist_ok=True)
    name = class_info["name"]
    ctype = class_info["type"]
    attributes = class_info.get("attributes", [])
    methods = class_info.get("methods", [])
    extends = class_info.get("extends")
    implements = class_info.get("implements", [])

    header = f"public {ctype} {name}"
    if extends and ctype == "class":
        header += f" extends {extends}"
    if implements and ctype == "class":
        header += f" implements {', '.join(implements)}"
    header += " {"

    lines = [header]

    # Enums: list constants
    if ctype == "enum":
        enum_constants = ", ".join([a['name'].upper() for a in attributes]) + ";"
        lines.append(f"    {enum_constants}")
    else:
        # Attributes
        for attr in attributes:
            lines.append(f"    private {attr['type']} {attr['name']};")

        lines.append("")

        # Constructor (skip for interface/enum)
        if ctype == "class" and attributes:
            params = ", ".join([f"{a['type']} {a['name']}" for a in attributes])
            lines.append(f"    public {name}({params}) {{")
            for a in attributes:
                lines.append(f"        this.{a['name']} = {a['name']};")
            lines.append("    }")
            lines.append("")

        # Getters & Setters (class only)
        if ctype == "class":
            for attr in attributes:
                n = attr["name"].capitalize()
                t = attr["type"]
                lines.append(f"    public {t} get{n}() {{ return this.{attr['name']}; }}")
                lines.append(f"    public void set{n}({t} {attr['name']}) {{ this.{attr['name']} = {attr['name']}; }}")
                lines.append("")

    # Methods (interfaces only contain declarations)
    for m in methods:
        params = ", ".join([f"{p['type']} {p['name']}" for p in m.get("parameters", [])])
        if ctype == "interface":
            lines.append(f"    {m['return_type']} {m['name']}({params});")
        else:
            lines.append(f"    public {m['return_type']} {m['name']}({params}) {{")
            lines.append("        // Use this, super, or instanceof as needed")
            lines.append("    }")
        lines.append("")

    lines.append("}")

    name = re.sub(r'[^A-Za-z0-9]+', '', name)

    filepath = os.path.join(output_dir, f"{name}.java")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def generate_from_image(image_path):
    a = str(image_path).split("\\")
    folder_name = os.path.splitext(a[len(a)-1])[0]

    folder_path = "..\\Dataset\\SC\\"+str(a[len(a)-3])+"\\labels\\"+folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    classes = extract_classes_from_image(image_path)

    save_as_json(classes)

    for c in classes:
        generate_java_class(c, folder_path)

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

# === RUN ===
if __name__ == "__main__":

    if not os.path.exists("..\\Dataset\\SC\\train\\labels"):
        sc_train_path = getListOfFiles("..\\Dataset\\SC\\train\\images")
        for x in range(len(sc_train_path)):
            print(sc_train_path[x])
            generate_from_image(sc_train_path[x])

    if not os.path.exists("..\\Dataset\\SC\\test\\labels"):
        sc_test_path = getListOfFiles("..\\Dataset\\SC\\test\\images")
        for x in range(len(sc_test_path)):
            print(sc_test_path[x])
            generate_from_image(sc_test_path[x])