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
_NUMBER_PREFIX = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+")
# Matches a TOC entry with dot/ellipsis leaders: "<title> ....... <page>".
_TOC_ENTRY_DOTS = re.compile(
    r"^\s*(?P<title>.+?)\s*(?:[.\u2026]\s*){2,}\s*(?P<page>\d+)\s*$"
)
# Matches a TOC entry without leaders: "<title>   <page>" (2+ spaces or tab).
_TOC_ENTRY_PLAIN = re.compile(
    r"^\s*(?P<title>.+?)(?:\s{2,}|\t+)(?P<page>\d+)\s*$"
)
# Fallback: any line beginning with a numeric prefix (e.g. "1.", "1.0",
# "1.11.2"), with an optional trailing page number separated by single space.
_TOC_ENTRY_NUMBERED = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)*\.?)\s+(?P<rest>.+?)\s*$"
)
_TRAILING_PAGE = re.compile(r"^(?P<title>.*?)\s+(?P<page>\d+)\s*$")
_TOC_HEADER = re.compile(r"^\s*(table of contents|contents)\s*$", re.IGNORECASE)


def _level_from_number(title: str) -> int:
    """Infer heading level from a numeric prefix like '1.', '1.1', '1.2.3'.

    Trailing ``.0`` components are treated as conventional placeholders
    rather than real depth — i.e. ``"1.0 Introduction"`` is level 1, not
    level 2, matching the printed-TOC convention where ``N.0`` denotes a
    top-level chapter.
    """
    m = _NUMBER_PREFIX.match(title)
    if not m:
        return 1
    parts = m.group("num").rstrip(".").split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return len(parts)


def _match_toc_entry(line: str) -> dict | None:
    """Match a TOC entry line.

    Returns a dict ``{"title", "page", "variant"}`` on success, where
    ``page`` may be ``None`` for numbered entries that lack a trailing
    page number (some printed TOCs put the page on the next line, or
    omit it for top-level chapter headings).

    Tried in order:
      1. dot-leader form: ``"Title ...... 12"``
      2. whitespace-separated form: ``"Title    12"`` (2+ spaces / tab)
      3. numbered fallback: any line starting with ``N(.N)*`` plus body,
         page optional (handles ``"1.0 Introduction"`` and
         ``"2.0 Seismic Data Processing 15"`` with a single space).
    """
    m = _TOC_ENTRY_DOTS.match(line)
    if m:
        return {"title": m.group("title").strip(), "page": m.group("page"), "variant": "dots"}

    m = _TOC_ENTRY_PLAIN.match(line)
    if m:
        title = m.group("title").strip()
        # Reject obvious body-text lines that happen to end with a number.
        ok = True
        if len(title) > 90:
            ok = False
        elif title.endswith((".", "!", "?", ":", ",", ";")) and not _NUMBER_PREFIX.match(title):
            ok = False
        if ok:
            return {"title": title, "page": m.group("page"), "variant": "plain"}

    # Numbered fallback — only for lines starting with a section number.
    m = _TOC_ENTRY_NUMBERED.match(line)
    if not m:
        return None
    num = m.group("num")
    rest = m.group("rest").strip()
    if not rest:
        return None

    # Split off a trailing integer as the page number, if present.
    page = None
    tail = _TRAILING_PAGE.match(rest)
    if tail:
        candidate = tail.group("title").strip()
        if candidate:
            rest = candidate
            page = tail.group("page")

    title = f"{num} {rest}".strip()
    # Guardrails so we don't eat numbered body sentences.
    if len(title) > 120:
        return None
    if rest.endswith((".", "!", "?", ":", ";")):
        return None
    return {"title": title, "page": page, "variant": "numbered"}


# ---------------------------------------------------------------------------
# pdfplumber-based TOC extraction (preferred — fast, no model load)
# ---------------------------------------------------------------------------

def _build_page_id_map(pdfminer_doc) -> dict:
    """Map pdfminer page object ids to 1-based page numbers."""
    from pdfminer.pdfpage import PDFPage
    return {id(page): idx for idx, page in enumerate(PDFPage.create_pages(pdfminer_doc), start=1)}


def _resolve_outline_page(dest, action, pdfminer_doc, page_id_map: dict):
    """Best-effort resolution of an outline destination to a 1-based page number."""
    target = None
    try:
        # Named destination -> resolve via document
        if isinstance(dest, (bytes, str)):
            try:
                resolved = pdfminer_doc.get_dest(dest)
            except Exception:
                resolved = None
            if resolved:
                target = resolved[0] if isinstance(resolved, list) else resolved
        elif isinstance(dest, list) and dest:
            target = dest[0]
        elif action and isinstance(action, dict):
            d = action.get("D")
            if isinstance(d, list) and d:
                target = d[0]

        if target is None:
            return None

        # Resolve indirect references
        if hasattr(target, "resolve"):
            target = target.resolve()

        return page_id_map.get(id(target))
    except Exception:
        return None


def _assign_outline_numbers(toc: list[dict]) -> list[dict]:
    """Prefix synthesized section numbers (1, 1.1, 1.1.1 …) onto outline
    entries that don't already carry a numeric prefix.

    The outline already encodes hierarchy via ``level``; we just walk the
    entries in order maintaining a per-level counter. When ``level`` jumps
    by more than 1 (e.g. 1 -> 3) we pad intermediate levels with zeros so
    the numbering still reflects depth.
    """
    counters: list[int] = []
    out: list[dict] = []
    for entry in toc:
        level = max(1, int(entry.get("level") or 1))

        # Pad / trim the counter stack to the current level.
        if len(counters) < level:
            counters.extend([0] * (level - len(counters)))
        else:
            counters = counters[:level]

        counters[level - 1] += 1
        number = ".".join(str(c) for c in counters)

        title = entry["title"]
        # Don't double-number titles that already start with "1.", "1.1", etc.
        if _NUMBER_PREFIX.match(title):
            new_title = title
        else:
            new_title = f"{number} {title}"

        out.append({**entry, "title": new_title, "number": number})
    return out


def _toc_from_pdf_outline(pdf_path: Path) -> list[dict]:
    """Extract TOC from a PDF's embedded outline (bookmarks) via pdfplumber."""
    logger.info("[TOC] Trying method: embedded PDF outline (pdfplumber)")
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; cannot read PDF outline.")
        return []

    toc: list[dict] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pdfminer_doc = pdf.doc
            try:
                outlines = list(pdfminer_doc.get_outlines())
            except Exception as exc:
                logger.info("PDF has no outline/bookmarks: %s", exc)
                return []

            if not outlines:
                return []

            page_id_map = _build_page_id_map(pdfminer_doc)
            for entry in outlines:
                # pdfminer yields (level, title, dest, action, se)
                level = entry[0] if len(entry) > 0 else 1
                title = entry[1] if len(entry) > 1 else ""
                dest = entry[2] if len(entry) > 2 else None
                action = entry[3] if len(entry) > 3 else None

                title_str = str(title).strip()
                if not title_str:
                    continue
                try:
                    level_int = int(level)
                except (TypeError, ValueError):
                    level_int = 1

                page = _resolve_outline_page(dest, action, pdfminer_doc, page_id_map)
                toc.append({"level": level_int, "title": title_str, "page": page})
    except Exception as exc:
        logger.warning("Failed to read PDF outline: %s", exc)
        return []

    if toc:
        toc = _assign_outline_numbers(toc)
        logger.info("Extracted %d entries from PDF outline.", len(toc))
    return toc


def _toc_from_content_scan(pdf_path: Path, max_pages: int = 10) -> list[dict]:
    """Scan the first few pages for a 'Table of Contents' section and parse it.

    Looks for entries with dot-leaders followed by a page number, e.g.:
        '1. Introduction ............... 5'
        '1.2.1 Energy Source ...........  9'

    Enable verbose tracing of every scanned line by setting
    ``TOC_SCAN_DEBUG=1`` in the environment (or by setting this module's
    logger to DEBUG level).
    """
    logger.info(
        "[TOC] Trying method: printed-TOC content scan (pdfplumber, first %d pages)",
        max_pages,
    )
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; cannot scan PDF for TOC.")
        return []

    debug = (
        os.environ.get("TOC_SCAN_DEBUG", "").lower() in ("1", "true", "yes", "on")
        or logger.isEnabledFor(logging.DEBUG)
    )

    def _trace(tag: str, page_no: int, line_no: int, text: str, extra: str = "") -> None:
        if not debug:
            return
        snippet = text if len(text) <= 120 else text[:117] + "…"
        suffix = f"  [{extra}]" if extra else ""
        logger.info("[TOC-scan] p%-3d L%-3d %-12s | %s%s", page_no, line_no, tag, snippet, suffix)

    toc: list[dict] = []
    in_toc = False
    blank_streak = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages[:max_pages], start=1):
                text = page.extract_text() or ""
                if not text:
                    _trace("empty-page", page_idx, 0, "(no extractable text)")
                    continue

                if debug and not in_toc:
                    _trace("page-start", page_idx, 0, f"({len(text.splitlines())} lines)")

                for line_idx, raw_line in enumerate(text.splitlines(), start=1):
                    line = raw_line.rstrip()
                    stripped = line.strip()

                    if not stripped:
                        if in_toc and toc:
                            blank_streak += 1
                            _trace("blank", page_idx, line_idx, "", f"streak={blank_streak}")
                            if blank_streak >= 3:
                                _trace("stop-blanks", page_idx, line_idx, "3 consecutive blanks; ending TOC scan")
                                return toc
                        continue
                    blank_streak = 0

                    if not in_toc:
                        if _TOC_HEADER.match(stripped):
                            in_toc = True
                            _trace("toc-header", page_idx, line_idx, stripped)
                        else:
                            _trace("pre-toc", page_idx, line_idx, stripped)
                        continue

                    # Inside a candidate TOC region
                    m = _match_toc_entry(line)
                    if not m:
                        if toc and len(stripped) > 100:
                            _trace(
                                "stop-paragraph",
                                page_idx,
                                line_idx,
                                stripped,
                                f"len={len(stripped)} > 100; ending TOC scan",
                            )
                            return toc
                        _trace("no-match", page_idx, line_idx, stripped, f"len={len(stripped)}")
                        continue

                    title = m["title"].strip().rstrip(".").strip()
                    if not title or _TOC_HEADER.match(title):
                        _trace("skip-header-echo", page_idx, line_idx, stripped)
                        continue
                    page_raw = m.get("page")
                    page_num: int | None = None
                    if page_raw is not None:
                        try:
                            page_num = int(page_raw)
                        except ValueError:
                            _trace("bad-pagenum", page_idx, line_idx, stripped)
                            page_num = None

                    level = _level_from_number(title)
                    toc.append(
                        {
                            "level": level,
                            "title": title,
                            "page": page_num,
                        }
                    )
                    _trace(
                        "match",
                        page_idx,
                        line_idx,
                        title,
                        f"variant={m.get('variant')} level={level} page={page_num}",
                    )
    except Exception as exc:
        logger.warning("pdfplumber content scan failed: %s", exc)
        return []

    if toc:
        logger.info("Extracted %d entries from in-document TOC page(s).", len(toc))
    elif debug:
        logger.info("[TOC-scan] no entries collected (in_toc=%s).", in_toc)
    return toc


def _toc_via_pdfplumber(pdf_path: Path) -> tuple[list[dict], str | None]:
    """Try outline first, fall back to scanning printed TOC pages.

    Returns (toc, method) where method is one of
    'embedded-outline', 'content-scan', or None when nothing was found.
    """
    outline_toc = _toc_from_pdf_outline(pdf_path)
    if outline_toc:
        return outline_toc, "embedded-outline"

    logger.info("[TOC] No usable PDF outline; falling back to printed-TOC scan")
    scan_toc = _toc_from_content_scan(pdf_path)
    if scan_toc:
        return scan_toc, "content-scan"
    return [], None


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
        logger.info("[TOC] docling sub-method: numbered-heading level detection")
        return _toc_from_numbered_headings(headings)
    else:
        logger.info("[TOC] docling sub-method: bounding-box font-size level detection")
        return _toc_from_size(headings)


def extract_toc(pdf_path: Path) -> list[dict]:
    """Extract TOC from a PDF.

    Resolution order:
      1. PDF embedded outline / bookmarks (pdfplumber).
      2. Printed TOC page(s) scanned from the first ~15 pages (pdfplumber).
      3. Docling structural heading detection (slower, model-backed).
    """
    toc, method = _toc_via_pdfplumber(pdf_path)
    if toc:
        logger.info("[TOC] Method used: %s (%d entries)", method, len(toc))
        return toc

    logger.info("[TOC] Falling back to method: docling structural heading detection")
    conv_result = _convert_with_docling(pdf_path)
    headings = _collect_headings(conv_result.document)

    if not headings:
        logger.warning("[TOC] No headings detected by any method.")
        return []

    toc = _headings_to_toc(headings)
    logger.info("[TOC] Method used: docling (%d entries)", len(toc))
    return toc


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
    """Extract TOC (pdfplumber-preferred) and optionally figures (docling)."""
    toc, method = _toc_via_pdfplumber(pdf_path)
    figures: dict[int, list[Path]] = {}

    if toc:
        logger.info("[TOC] Method used: %s (%d entries)", method, len(toc))

    needs_docling = not toc or extract_images
    if not needs_docling:
        return toc, figures

    logger.info("Running docling standard pipeline (single pass) …")
    conv_result = _convert_with_docling(pdf_path, extract_images=extract_images)

    if not toc:
        logger.info("[TOC] Falling back to method: docling structural heading detection")
        headings = _collect_headings(conv_result.document)
        if headings:
            toc = _headings_to_toc(headings)
            logger.info("[TOC] Method used: docling (%d entries)", len(toc))
        else:
            logger.warning("[TOC] No headings detected by any method.")

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
                        # Skip blank lines in the TOC file.
                        while toc_line and not toc_line.strip():
                            toc_line = toc_file.readline()

                        # TOC exhausted before the markdown ran out of
                        # headings — fall back to markdown-heading mode for
                        # the rest of the document instead of warning on
                        # every remaining heading.
                        if not toc_line:
                            use_markdown_headings = True
                            flush_text()
                            attach_node(heading_level, heading)
                            previous_line = line
                            line = markdown_file.readline()
                            continue

                        parts = toc_line.split(";", 1)
                        if len(parts) < 2:
                            logger.warning(f"Invalid TOC line format: {toc_line!r}")
                            toc_line = toc_file.readline()
                            continue

                        level, heading_toc = parts
                        if (SequenceMatcher(None, "contents", heading_toc.lower())).ratio() > 0.6:
                            toc_line = toc_file.readline()
                            while toc_line and not toc_line.strip():
                                toc_line = toc_file.readline()
                            if not toc_line:
                                use_markdown_headings = True
                                flush_text()
                                attach_node(heading_level, heading)
                                previous_line = line
                                line = markdown_file.readline()
                                continue
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

            # Always flush whatever remains for the last section, regardless of
            # how the loop terminated (EOF reached either via the bottom of the
            # loop or via the blank-line `continue` branch above).
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
