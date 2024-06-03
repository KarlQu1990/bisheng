import fitz
import logging
import os
import re

from langchain.docstore.document import Document
from langchain.document_loaders.pdf import BasePDFLoader
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage
from typing import Dict, List, Tuple
from pymupdf4llm.helpers.get_text_lines import get_raw_lines, is_white
from pymupdf4llm.helpers.multi_column import column_boxes

logger = logging.getLogger(__name__)


bullet = ("- ", "* ", chr(0xF0A7), chr(0xF0B7), chr(0xB7), chr(8226), chr(9679))
GRAPHICS_TEXT = "\n![%s](%s)\n"


class IdentifyHeaders:
    """Compute data for identifying header text."""

    def __init__(
        self,
        doc: str,
        pages: list = None,
        body_limit: float = 12,
    ):
        """Read all text and make a dictionary of fontsizes.

        Args:
            pages: optional list of pages to consider
            body_limit: consider text with larger font size as some header
        """
        if isinstance(doc, fitz.Document):
            mydoc = doc
        else:
            mydoc = fitz.open(doc)

        if pages is None:  # use all pages if omitted
            pages = range(mydoc.page_count)

        fontsizes = {}
        for pno in pages:
            page = mydoc.load_page(pno)
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for span in [  # look at all non-empty horizontal spans
                s for b in blocks for line in b["lines"] for s in line["spans"] if not is_white(s["text"])
            ]:
                fontsz = round(span["size"])
                count = fontsizes.get(fontsz, 0) + len(span["text"].strip())
                fontsizes[fontsz] = count

        if mydoc != doc:
            # if opened here, close it now
            mydoc.close()

        # maps a fontsize to a string of multiple # header tag characters
        self.header_id = {}

        # If not provided, choose the most frequent font size as body text.
        # If no text at all on all pages, just use 12.
        # In any case all fonts not exceeding
        temp = sorted(
            [(k, v) for k, v in fontsizes.items()],
            key=lambda i: i[1],
            reverse=True,
        )
        if temp:
            b_limit = max(body_limit, temp[0][0])
        else:
            b_limit = body_limit

        # identify up to 6 font sizes as header candidates
        sizes = sorted(
            [f for f in fontsizes.keys() if f > b_limit],
            reverse=True,
        )[:6]

        # make the header tag dictionary
        for i, size in enumerate(sizes):
            self.header_id[size] = "#" * (i + 1) + " "

    def get_header_id(self, span: dict, page=None) -> str:
        """Return appropriate markdown header prefix.

        Given a text span from a "dict"/"rawdict" extraction, determine the
        markdown header prefix string of 0 to n concatenated '#' characters.
        """
        fontsize = round(span["size"])  # compute fontsize
        hdr_id = self.header_id.get(fontsize, "")
        return hdr_id


def to_markdown(
    doc: str,
    *,
    pages: list = None,
    hdr_info=None,
    write_images: bool = False,
    page_chunks: bool = False,
    margins=(0, 50, 0, 50),
) -> str:
    """Process the document and return the text of its selected pages."""

    if isinstance(doc, str):
        doc = fitz.open(doc)

    if pages is None:  # use all pages if no selection given
        pages = list(range(doc.page_count))

    if hasattr(margins, "__float__"):
        margins = [margins] * 4
    if len(margins) == 2:
        margins = (0, margins[0], 0, margins[1])
    if len(margins) != 4:
        raise ValueError("margins must have length 2 or 4 or be a number.")
    elif not all([hasattr(m, "__float__") for m in margins]):
        raise ValueError("margin values must be numbers")

    # If "hdr_info" is not an object having method "get_header_id", scan the
    # document and use font sizes as header level indicators.
    if callable(hdr_info):
        get_header_id = hdr_info
    elif hasattr(hdr_info, "get_header_id") and callable(hdr_info.get_header_id):
        get_header_id = hdr_info.get_header_id
    else:
        hdr_info = IdentifyHeaders(doc)
        get_header_id = hdr_info.get_header_id

    def resolve_links(links, span):
        """Accept a span and return a markdown link string."""
        bbox = fitz.Rect(span["bbox"])  # span bbox
        # a link should overlap at least 70% of the span
        bbox_area = 0.7 * abs(bbox)
        for link in links:
            hot = link["from"]  # the hot area of the link
            if not abs(hot & bbox) >= bbox_area:
                continue  # does not touch the bbox
            text = f'[{span["text"].strip()}]({link["uri"]})'
            return text

    def save_image(page, rect, i):
        """Optionally render the rect part of a page."""
        filename = page.parent.name.replace("\\", "/")
        image_path = f"{filename}-{page.number}-{i}.png"
        if write_images is True:
            pix = page.get_pixmap(clip=rect)
            pix.save(image_path)
            del pix
            return os.path.basename(image_path)
        return ""

    def write_text(
        page: fitz.Page,
        textpage: fitz.TextPage,
        clip: fitz.Rect,
        tabs=None,
        tab_rects: dict = None,
        img_rects: dict = None,
        links: list = None,
    ) -> Tuple[str, Dict[int, str], Dict[str, str]]:
        """Output the text found inside the given clip.

        This is an alternative for plain text in that it outputs
        text enriched with markdown styling.
        The logic is capable of recognizing headers, body text, code blocks,
        inline code, bold, italic and bold-italic styling.
        There is also some effort for list supported (ordered / unordered) in
        that typical characters are replaced by respective markdown characters.

        'tab_rects'/'img_rects' are dictionaries of table, respectively image
        or vector graphic rectangles.
        General Markdown text generation skips these areas. Tables are written
        via their own 'to_markdown' method. Images and vector graphics are
        optionally saved as files and pointed to by respective markdown text.
        """
        if clip is None:
            clip = textpage.rect

        out_string = ""
        tab_strings = {}
        img_strings = {}

        # This is a list of tuples (linerect, spanlist)
        nlines = get_raw_lines(textpage, clip=clip, tolerance=3)

        tab_rects0 = list(tab_rects.values())
        img_rects0 = list(img_rects.values())

        prev_lrect = None  # previous line rectangle
        prev_bno = -1  # previous block number of line
        code = False  # mode indicator: outputting code
        prev_hdr_string = None

        def get_sorted_rects(tab_or_img_rects, lrect) -> List[Tuple[int, fitz.Rect]]:
            # Pick up tables intersecting this text block
            return sorted(
                [j for j in tab_or_img_rects.items() if j[1].y1 <= lrect.y0 and not (j[1] & clip).is_empty],
                key=lambda j: (j[1].y1, j[1].x0),
            )

        for lrect, spans in nlines:
            # there may tables or images inside the text block: skip them
            if intersects_rects(lrect, tab_rects0) or intersects_rects(lrect, img_rects0):
                continue

            # Pick up tables intersecting this text block
            sorted_tab_rects = get_sorted_rects(tab_rects, lrect)
            for i, tab_rect in sorted_tab_rects:
                tab_strings[i] = tabs[i].to_markdown(clean=False)
                del tab_rects[i]

            # Pick up images / graphics intersecting this text block
            sorted_tab_rects = get_sorted_rects(img_rects, lrect)
            for i, img_rect in sorted_tab_rects:
                pathname = save_image(page, img_rect, i)
                if pathname:
                    img_strings[i] = GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]

            text = " ".join([s["text"] for s in spans])

            # if the full line mono-spaced?
            all_mono = all([s["flags"] & 8 for s in spans])

            if all_mono:
                if not code:  # if not already in code output  mode:
                    out_string += "```\n"  # switch on "code" mode
                    code = True
                # compute approx. distance from left - assuming a width
                # of 0.5*fontsize.
                delta = int((lrect.x0 - clip.x0) / (spans[0]["size"] * 0.5))
                indent = " " * delta

                out_string += indent + text + "\n"
                continue  # done with this line

            span0 = spans[0]
            bno = span0["block"]  # block number of line
            if bno != prev_bno:
                out_string += "\n"
                prev_bno = bno

            if (  # check if we need another line break
                prev_lrect
                and lrect.y1 - prev_lrect.y1 > lrect.height * 1.5
                or span0["text"].startswith("[")
                or span0["text"].startswith(bullet)
                or span0["flags"] & 1  # superscript?
            ):
                out_string += "\n"
            prev_lrect = lrect

            # if line is a header, this will return multiple "#" characters
            hdr_string = get_header_id(span0, page=page)

            # intercept if header text has been broken in multiple lines
            if hdr_string and hdr_string == prev_hdr_string:
                out_string = out_string[:-1] + " " + text + "\n"
                continue

            prev_hdr_string = hdr_string
            if hdr_string.startswith("#"):  # if a header line skip the rest
                out_string += hdr_string + text + "\n"
                continue

            # this line is not all-mono, so switch off "code" mode
            if code:  # still in code output mode?
                out_string += "```\n"  # switch of code mode
                code = False

            for i, s in enumerate(spans):  # iterate spans of the line
                # decode font properties
                mono = s["flags"] & 8
                bold = s["flags"] & 16
                italic = s["flags"] & 2

                if mono:
                    # this is text in some monospaced font
                    out_string += f"`{s['text'].strip()}` "
                else:  # not a mono text
                    prefix = ""
                    suffix = ""
                    if hdr_string == "":
                        if bold:
                            prefix = "**"
                            suffix += "**"
                        if italic:
                            prefix += "_"
                            suffix = "_" + suffix

                    # convert intersecting link into markdown syntax
                    ltext = resolve_links(links, s)
                    if ltext:
                        text = f"{hdr_string}{prefix}{ltext}{suffix} "
                    else:
                        text = f"{hdr_string}{prefix}{s['text'].strip()}{suffix} "

                    if text.startswith(bullet):
                        text = "-  " + text[1:]
                    out_string += text

            if not code:
                out_string += "\n"

        out_string += "\n"
        if code:
            out_string += "```\n"  # switch of code mode
            code = False

        out_string = re.sub(r"(\s)+(\n)(\s)+", "\n", out_string)

        return out_string, tab_strings, img_strings

    def is_in_rects(rect, rect_list):
        """Check if rect is contained in a rect of the list."""
        for i, r in enumerate(rect_list, start=1):
            if rect in r:
                return i
        return 0

    def intersects_rects(rect, rect_list):
        """Check if middle of rect is contained in a rect of the list."""
        for i, r in enumerate(rect_list, start=1):
            if (rect.tl + rect.br) / 2 in r:  # middle point is inside r
                return i
        return 0

    def get_sorted_rects(tab_or_img_rects, text_rect=None) -> List[Tuple[int, fitz.Rect]]:
        if text_rect is not None:
            # select tables above the text block
            return sorted(
                [j for j in tab_or_img_rects.items() if j[1].y1 <= text_rect.y0], key=lambda j: (j[1].y1, j[1].x0)
            )

        # output all remaining table
        return sorted(tab_or_img_rects.items(), key=lambda j: (j[1].y1, j[1].x0))

    def output_tables(tabs, text_rect, tab_rects) -> Dict[int, str]:
        """Output tables above a text rectangle."""
        this_mds = {}  # markdown string for table content

        sorted_rects = get_sorted_rects(tab_rects, text_rect)
        for i, trect in sorted_rects:
            this_mds[i] = tabs[i].to_markdown(clean=False)
            del tab_rects[i]  # do not touch this table twice

        return this_mds

    def output_images(page, text_rect, img_rects) -> Dict[int, str]:
        """Output images and graphics above text rectangle."""
        if img_rects is None:
            return {}

        this_mds = {}  # markdown string
        sorted_rects = get_sorted_rects(img_rects, text_rect)
        for i, img_rect in sorted_rects:
            pathname = save_image(page, img_rect, i)
            if pathname:
                this_mds[i] = GRAPHICS_TEXT % (pathname, pathname)
            del img_rects[i]  # do not touch this image twice

        return this_mds

    def get_metadata(doc, pno):
        meta = doc.metadata.copy()
        meta["file_path"] = doc.name
        meta["page_count"] = doc.page_count
        meta["page"] = pno + 1
        return meta

    def get_page_output(doc, pno, margins, textflags):
        """Process one page.

        Args:
            doc: fitz.Document
            pno: 0-based page number
            textflags: text extraction flag bits

        Returns:
            Markdown string of page content and image, table and vector
            graphics information.
        """
        page = doc[pno]
        md_string = ""
        left, top, right, bottom = margins
        clip = page.rect + (left, top, -right, -bottom)
        # extract all links on page
        links = [link for link in page.get_links() if link["kind"] == 2]

        # make a TextPage for all later extractions
        textpage = page.get_textpage(flags=textflags, clip=clip)

        img_info = [img for img in page.get_image_info() if img["bbox"] in clip]
        images = img_info[:]
        tables = []
        graphics = []

        # Locate all tables on page
        tabs = page.find_tables(clip=clip, strategy="lines_strict")

        # Make a list of table boundary boxes.
        # Must include the header bbox (may exist outside tab.bbox)
        tab_rects = {}
        for i, t in enumerate(tabs):
            tab_rects[i] = fitz.Rect(t.bbox) | fitz.Rect(t.header.bbox)
            tab_dict = {
                "bbox": tuple(tab_rects[i]),
                "rows": t.row_count,
                "columns": t.col_count,
            }
            tables.append(tab_dict)
        tab_rects0 = list(tab_rects.values())

        # Select paths that are not contained in any table
        page_clip = page.rect + (36, 36, -36, -36)  # ignore full page graphics
        paths = [
            p
            for p in page.get_drawings()
            if not intersects_rects(p["rect"], tab_rects0)
            and p["rect"] in page_clip
            and p["rect"].width < page_clip.width
            and p["rect"].height < page_clip.height
        ]

        # Determine vector graphics outside any tables, filerting out any
        # which contain no stroked paths
        vg_clusters = []
        for bbox in page.cluster_drawings(drawings=paths):
            include = False
            for p in [p for p in paths if p["rect"] in bbox]:
                if p["type"] != "f":
                    include = True
                    break
                if [item[0] for item in p["items"] if item[0] == "c"]:
                    include = True
                    break
                if include is True:
                    vg_clusters.append(bbox)

        actual_paths = [p for p in paths if is_in_rects(p["rect"], vg_clusters)]

        vg_clusters0 = [r for r in vg_clusters if not intersects_rects(r, tab_rects0) and r.height > 20]

        if write_images is True:
            vg_clusters0 += [fitz.Rect(i["bbox"]) for i in img_info]

        vg_clusters = dict((i, r) for i, r in enumerate(vg_clusters0))

        # Determine text column bboxes on page, avoiding tables and graphics
        text_rects = column_boxes(
            page,
            paths=actual_paths,
            no_image_text=write_images,
            textpage=textpage,
            avoid=tab_rects0 + vg_clusters0,
        )
        """Extract markdown text iterating over text rectangles.
        We also output any tables. They may live above, below or inside
        the text rectangles.
        """
        tab_strings = {}
        img_strings = {}

        for text_rect in text_rects:
            # output tables above this block of text
            tab_string = output_tables(tabs, text_rect, tab_rects)
            img_string = output_images(page, text_rect, vg_clusters)
            tab_strings.update(tab_string)
            img_strings.update(img_string)

            # output text inside this rectangle
            text_string, tab_string, img_string = write_text(
                page,
                textpage,
                text_rect,
                tabs=tabs,
                tab_rects=tab_rects,
                img_rects=vg_clusters,
                links=links,
            )
            md_string += text_string
            tab_strings.update(tab_string)
            img_string.update(img_string)

        # write any remaining tables and images
        tab_string = output_tables(tabs, None, tab_rects)
        img_string = output_images(None, tab_rects, None)
        tab_strings.update(tab_string)
        img_strings.update(img_string)

        while md_string.startswith("\n"):
            md_string = md_string[1:]

        for i, tab in enumerate(tables):
            tab_string = tab_strings.get(i, None)
            if tab_string:
                tab["text"] = tab_string

        for i, img in enumerate(graphics):
            img_string = img_strings.get(i, None)
            if img_string:
                img["text"] = img_string

        md_string = re.sub(r"(\s){2,}", " ", md_string)
        md_string = re.sub(r"(\n){3,}", "\n\n", md_string)

        return md_string, images, tables, graphics

    if page_chunks is False:
        document_output = ""
    else:
        document_output = []

    # read the Table of Contents
    toc = doc.get_toc()
    textflags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_MEDIABOX_CLIP
    for pno in pages:
        page_output, images, tables, graphics = get_page_output(doc, pno, margins, textflags)
        if page_chunks is False:
            document_output += page_output
        else:
            # build subet of TOC for this page
            page_tocs = [t for t in toc if t[-1] == pno + 1]

            metadata = get_metadata(doc, pno)
            document_output.append(
                {
                    "metadata": metadata,
                    "toc_items": page_tocs,
                    "tables": tables,
                    "images": images,
                    "graphics": graphics,
                    "text": page_output,
                }
            )

    return document_output


class CustomPDFLoader(BasePDFLoader):
    """自定义pdf加载器。使用PyMuPDF读取pdf文本和表格，使用LLM总结表格内容。"""

    _prompt_fmt = """你是一个表格总结助手，请详细总结如下表格内容：
    {content}"""

    def __init__(
        self,
        file_path: str,
        llm: BaseLanguageModel,
    ) -> None:
        super().__init__(file_path)
        self.llm = llm

    def load(self) -> List[Document]:
        results = to_markdown(self.file_path, page_chunks=True)
        text_docs: List[Document] = []
        tab_docs: List[Document] = []

        for res in results:
            file_path = res["metadata"]["file_path"]
            page = res["metadata"]["page"]

            metadata = {"source": file_path, "page": page, "raw": "", "type": "text"}
            text_docs.append(Document(page_content=res["text"], metadata=metadata))

            for tab in res["tables"]:
                tab_docs.append(
                    Document(
                        page_content="",
                        metadata={"source": self.file_path, "page": page, "raw": tab["text"], "type": "table"},
                    )
                )

        # 过滤无效页数据
        i = 0
        n = 20
        num_docs = len(text_docs)
        while i < num_docs:
            page_content = text_docs[i].page_content
            head = page_content[:-n]
            tail = page_content[-n:]

            tail = re.sub(r"(-[\s]+[\d]+[\s]+-)|(第[\s]+[\d]+[\s]+页)|(共[\s]+[\d]+[\s]+页)", "", tail)
            text_docs[i].page_content = page_content = head + tail

            if num_docs <= 1:
                break

            if not page_content:
                text_docs.pop(i)
                num_docs = len(text_docs)
                continue

            if len(page_content) >= n:
                i += 1
                continue

            if i == 0:
                curr_doc = text_docs[i]
                next_doc = text_docs[i + 1]

                doc = Document(page_content=page_content + next_doc.page_content, metadata=curr_doc.metadata)
                text_docs[i + 1] = doc
                for j in range(i + 2, num_docs):
                    text_docs[j].metadata["page"] -= 1
            else:
                prev_doc = text_docs[i - 1]
                curr_doc = text_docs[i]

                doc = Document(page_content=prev_doc.page_content + page_content, metadata=prev_doc.metadata)
                text_docs[i - 1] = doc
                for j in range(i + 1, num_docs):
                    text_docs[j].metadata["page"] -= 1

            text_docs.pop(i)
            num_docs = len(text_docs)

        if not tab_docs:
            return text_docs

        prompts = [self._prompt_fmt.format(content=doc.metadata["raw"]) for doc in tab_docs]
        messages: List[BaseMessage] = self.llm.batch(prompts)

        for i, msg in enumerate(messages):
            tab_docs[i].page_content = msg.content

        return text_docs + tab_docs
