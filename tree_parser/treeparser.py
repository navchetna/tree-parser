import re
import json 
import os
import logging
from difflib import SequenceMatcher
from pathlib import Path

from sortedcontainers import SortedDict

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.document import DocItemLabel
from docling_core.types.doc import PictureItem

from tree_parser.node import Node
from tree_parser.text import Text
from tree_parser.table import Table
from tree_parser.utils import mkdirIfNotExists


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

_LEVEL_PATTERN = re.compile(r"^\d+(\.\d+)*\.?\s")


def _convert_with_docling(pdf_path: Path, extract_images: bool = False):
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
    )
    if extract_images:
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )
    result = doc_converter.convert(pdf_path)
    assert result.status == ConversionStatus.SUCCESS, f"Docling conversion failed: {result.status}"
    return result


def _collect_headings(doc) -> list[dict]:
    heading_labels = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
    headings = []

    for item, _ in doc.iterate_items():
        if item.label not in heading_labels:
            continue

        page = item.prov[0].page_no if item.prov else None

        bbox = item.prov[0].bbox if item.prov else None
        height = None
        if bbox is not None:
            try:
                height = abs(bbox.t - bbox.b)
            except AttributeError:
                try:
                    coords = list(bbox)
                    height = abs(coords[2][1] - coords[0][1])
                except Exception:
                    height = None

        headings.append({
            "title": item.text.replace("\n", " "),
            "page": page,
            "height": height,
        })

    return headings


def _toc_from_numbered_headings(headings: list[dict]) -> list[dict]:
    toc = []
    for h in headings:
        if not _LEVEL_PATTERN.match(h["title"]):
            continue
        number, _ = h["title"].split(" ", 1)
        level = number.count(".") + 1
        toc.append({"level": level, "title": h["title"], "page": h.get("page")})
    return toc


def _toc_from_size(headings: list[dict]) -> list[dict]:
    dict_level: SortedDict = SortedDict()
    list_headings = []

    for h in headings:
        if h["height"] is None:
            continue

        size = round(h["height"])
        idx = -1
        prev_level = 0
        size_lesser_found = False

        for key in reversed(dict_level):
            if size == key or size - 1 == key:
                idx = key
                break
            if size - 1 > key:
                prev_level = dict_level[key] - 1
                size_lesser_found = True
                break
            prev_level = dict_level[key]

        if size_lesser_found:
            for key in reversed(dict_level):
                if size - 1 > key:
                    dict_level[key] += 1
            for entry in list_headings:
                if size - 1 > entry[0]:
                    entry[1] += 1

        if idx == -1:
            idx = size
            dict_level[size] = prev_level + 1

        list_headings.append([idx, dict_level[idx], h["title"], h.get("page")])

    return [{"level": e[1], "title": e[2], "page": e[3]} for e in list_headings]


def _headings_to_toc(headings: list[dict]) -> list[dict]:
    if not headings:
        return []
    if any(_LEVEL_PATTERN.match(h["title"]) for h in headings):
        logger.info("Using numbered-heading level detection.")
        return _toc_from_numbered_headings(headings)
    else:
        logger.info("Using bounding-box size level detection.")
        return _toc_from_size(headings)


def extract_toc(pdf_path: Path) -> list[dict]:
    """Extract TOC from a PDF (lightweight single-purpose call)."""
    logger.info("Running docling standard pipeline to extract headings \u2026")
    conv_result = _convert_with_docling(pdf_path)
    headings = _collect_headings(conv_result.document)

    if not headings:
        logger.warning("No headings detected by docling.")
        return []

    return _headings_to_toc(headings)


def save_toc(toc: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in toc:
            f.write(f"{entry['level']};{entry['title']}\n")
    logger.info("Saved TOC to %s", output_path)


def extract_figures(pdf_path: Path, output_dir: Path, conv_result=None) -> dict[int, list[Path]]:
    """Extract all figures/pictures from a PDF. Uses conv_result if provided, otherwise runs a new conversion."""
    if conv_result is None:
        logger.info("Running standard pipeline for figure extraction \u2026")
        conv_result = _convert_with_docling(pdf_path, extract_images=True)

    saved: dict[int, list[Path]] = {}
    picture_counter = 0

    for element, _level in conv_result.document.iterate_items():
        if not isinstance(element, PictureItem):
            continue
        if not element.prov:
            continue

        page_no = element.prov[0].page_no
        picture_counter += 1

        pil_img = element.get_image(conv_result.document)
        if pil_img is None or pil_img.width == 0 or pil_img.height == 0:
            logger.debug("Skipping empty image for picture %d on page %d", picture_counter, page_no)
            continue

        page_dir = output_dir / "figures" / f"page_{page_no}"
        page_dir.mkdir(parents=True, exist_ok=True)

        img_path = page_dir / f"figure_{picture_counter}.png"
        with img_path.open("wb") as fp:
            pil_img.save(fp, "PNG")

        saved.setdefault(page_no, []).append(img_path)
        logger.info("Saved figure (page %d) \u2192 %s", page_no, img_path)

    logger.info("Saved %d figure(s) across %d page(s)", sum(len(v) for v in saved.values()), len(saved))
    return saved


def process_pdf(pdf_path: Path, output_dir: Path, extract_images: bool = True) -> tuple[list[dict], dict[int, list[Path]]]:
    """Single docling pass: extract TOC and optionally figures from a PDF."""
    logger.info("Running docling standard pipeline (single pass) \u2026")
    conv_result = _convert_with_docling(pdf_path, extract_images=extract_images)

    headings = _collect_headings(conv_result.document)
    toc = _headings_to_toc(headings) if headings else []
    if not headings:
        logger.warning("No headings detected by docling.")

    figures: dict[int, list[Path]] = {}
    if extract_images:
        figures = extract_figures(pdf_path, output_dir, conv_result=conv_result)

    return toc, figures


class TreeParser:
    def __init__(self, user_param: str, output_dir: str = None):
        # User-specific output directory
        self.OUTPUT_DIR = os.path.join(output_dir if output_dir else os.path.expanduser("~"), user_param)
        mkdirIfNotExists(self.OUTPUT_DIR)
        print(f"Initialized TreeParser with output directory: {self.OUTPUT_DIR}")

    def get_filename(self, file):
        return os.path.splitext(os.path.basename(file))[0]

    def peek_next_lines(self, f):
        pos = f.tell()
        line = f.readline()
        line_2 = f.readline()
        f.seek(pos)
        return line, line_2

    def parse_markdown(self, filename, rootNode, recentNodeDict):
        toc_file = open(os.path.join(self.OUTPUT_DIR, filename, "toc.txt"), "r")
        toc_line = toc_file.readline()
        use_markdown_headings = not bool(toc_line)
                
        currNode = rootNode
        content = ""
        previous_line = ""

        def level_value(level):
            try:
                return int(level)
            except (TypeError, ValueError):
                return 0

        def flush_text():
            nonlocal content
            if content:
                currNode.append_content(Text(content, currNode))
                content = ""

        def attach_node(level, heading):
            nonlocal currNode

            node_level = level_value(level)
            node = Node(node_level, heading, os.path.join(self.OUTPUT_DIR, filename))

            parent_level = 0
            for existing_level in sorted(recentNodeDict.keys(), reverse=True):
                if level_value(existing_level) < node_level:
                    parent_level = existing_level
                    break

            parent = recentNodeDict[parent_level]
            parent.append_child(node)
            node.set_parent(parent)

            for existing_level in list(recentNodeDict.keys()):
                if level_value(existing_level) >= node_level:
                    del recentNodeDict[existing_level]
            recentNodeDict[node_level] = node
            currNode = node

        with open(os.path.join(self.OUTPUT_DIR, filename, filename + ".md"), 'r') as markdown_file:
            line = markdown_file.readline()
            while line:
                line = re.sub(r'<span[^>]*?\/?>(</span>)?', '', line)
                if line == "\n":
                    line = markdown_file.readline()
                    continue
                # if bool(re.match(r'^#+', line)):
                #     _, heading = line.split(" ", 1)
                heading_regex = re.compile(r'^(#{1,6})\s+(.+)')
                match = heading_regex.match(line)
                if match:
                    heading_level = level_value(len(match.group(1)))
                    heading_text = match.group(2).strip()
                    heading = heading_text.strip().replace("*", "")

                    if use_markdown_headings:
                        flush_text()
                        attach_node(heading_level, heading)
                    else:
                        parts = toc_line.split(";", 1)
                        if len(parts) < 2:
                            logger.warning(f"Invalid TOC line format: {toc_line}")
                            toc_line = toc_file.readline()
                            continue

                        level, heading_toc = parts
                        if (SequenceMatcher(None, "contents", heading_toc.lower())).ratio() > 0.6:
                            toc_line = toc_file.readline()
                            parts = toc_line.split(";", 1)
                            if len(parts) < 2:
                                logger.warning(f"Invalid TOC line format after contents skip: {toc_line!r}")
                                toc_line = toc_file.readline()
                                continue

                            level, heading_toc = parts

                        if SequenceMatcher(None, heading.lower(), heading_toc.lower()).ratio() > 0.6:
                            flush_text()
                            attach_node(level, heading)
                            toc_line = toc_file.readline()
                        else:
                            content += line
                            previous_line = line
                            line = markdown_file.readline()
                            continue
                elif line[0] == '|':
                    flush_text()
                    table_list = [line]
                    while self.peek_next_lines(markdown_file)[0] and self.peek_next_lines(markdown_file)[0][0] == '|':
                        line = markdown_file.readline()
                        table_list.append(line)
                    next_line = self.peek_next_lines(markdown_file)[1].split('>', 1)
                    if len(next_line) > 1:
                        next_line = next_line[1]
                    else:
                        next_line = next_line[0]
                    pattern_table_heading = re.compile(r'^(Table|Figure)\s+(\d+)', re.IGNORECASE) 
                    match_table_heading_previous = pattern_table_heading.search(previous_line)
                    match_table_heading_next = pattern_table_heading.search(next_line)
                    heading = ""
                    if match_table_heading_previous:
                        heading = previous_line
                    elif match_table_heading_next:
                        heading = next_line
                    table_obj = Table("".join(table_list), heading, currNode)
                    currNode.append_content(table_obj)
                else:
                    pattern_heading = re.compile(r'^(Table|Figure)\s+(\d+)', re.IGNORECASE)
                    match_heading = pattern_heading.search(line)
                    if not match_heading:
                        content += line
                previous_line = line
                line = markdown_file.readline()
                if not line:
                    flush_text()

        if toc_file.readline():
            logger.warning("PDF not parsed accurately")

    def traverse_tree_text(self, node):
        if node is None:
            return
        
        node.output_node_info()
        total = node.get_length_children()
        for i in range(total):
            self.traverse_tree_text(node.get_child(i))
        
    def generate_output_text(self, tree):
        filename = self.get_filename(tree.file)
        output_path = os.path.join(self.OUTPUT_DIR, filename, "output.txt")
        with open(output_path, "w") as f:
            f.write("")
        self.traverse_tree_text(tree.rootNode)
        return output_path
    
    def traverse_tree_json(self, node):
        if node is None:
            return
        
        data = {}
        heading = node.get_heading()
        data[heading] = {}
        data[heading]['content'] = []

        content = node.get_content()
        for item in content:
            if isinstance(item, Text):
                data[heading]['content'].append({
                    'type': 'text',
                    'content': item.content
                })
            if isinstance(item, Table):
                data[heading]['content'].append({
                    'type': 'table',
                    'content': item.markdown_content
                })

        data[heading]['children'] = []
        
        total = node.get_length_children()
        for i in range(total):
            data[heading]['children'].append(self.traverse_tree_json(node.get_child(i)))
        
        return data

    def generate_output_json(self, tree):
        data = self.traverse_tree_json(tree.rootNode)
        filename = self.get_filename(tree.file)
        output_path = os.path.join(self.OUTPUT_DIR, filename, "output_tree.json")

        with open(output_path, "w") as outfile: 
            json.dump(data, outfile)
        logger.info(f"Saved JSON tree to {output_path}")
        return output_path

    def populate_tree(self, tree, toc: list[dict] = None, extract_images: bool = False):
        rootNode = tree.rootNode
        file = tree.file
        filename = self.get_filename(file)
        file_dir = Path(os.path.join(self.OUTPUT_DIR, filename))

        if toc is None:
            if extract_images:
                toc, _ = process_pdf(Path(file), file_dir, extract_images=True)
            else:
                toc = extract_toc(Path(file))

        save_toc(toc, file_dir / 'toc.txt')

        recentNodeDict = {}
        recentNodeDict[0] = rootNode

        self.parse_markdown(filename, rootNode, recentNodeDict)
    
    def get_output_path(self, tree):
        filename = self.get_filename(tree.file)
        return os.path.join(self.OUTPUT_DIR, filename, "output.txt")
