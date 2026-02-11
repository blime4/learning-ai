"""Generate a full-content PDF of the llama.cpp reading order guide.

Each file's actual content is included after its TOC entry, with markdown
rendered as styled paragraphs and source code rendered in monospace.
"""

import os
import re
import xml.sax.saxutils as saxutils

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Preformatted,
    KeepTogether,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
_FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
pdfmetrics.registerFont(TTFont("CJK", _FONT_PATH))

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
DARK = HexColor("#1a1a2e")
BLUE = HexColor("#0f3460")
LIGHT_BG = HexColor("#f0f0f5")
CODE_BG = HexColor("#f5f5f0")

# ---------------------------------------------------------------------------
# Base directory
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
s_title = ParagraphStyle(
    "Title", fontName="CJK", fontSize=22, leading=30,
    textColor=DARK, spaceAfter=6 * mm, alignment=1,
)
s_subtitle = ParagraphStyle(
    "Subtitle", fontName="CJK", fontSize=10, leading=14,
    textColor=HexColor("#666666"), spaceAfter=10 * mm, alignment=1,
)
s_overview = ParagraphStyle(
    "Overview", fontName="CJK", fontSize=9, leading=15,
    textColor=DARK, spaceBefore=4 * mm, spaceAfter=4 * mm,
    borderColor=BLUE, borderWidth=1, borderPadding=8, backColor=LIGHT_BG,
)
s_phase = ParagraphStyle(
    "Phase", fontName="CJK", fontSize=14, leading=20,
    textColor=BLUE, spaceBefore=6 * mm, spaceAfter=3 * mm,
)
s_file_heading = ParagraphStyle(
    "FileHeading", fontName="CJK", fontSize=11, leading=16,
    textColor=HexColor("#0f3460"), spaceBefore=6 * mm, spaceAfter=1 * mm,
)
s_file_desc = ParagraphStyle(
    "FileDesc", fontName="CJK", fontSize=8, leading=12,
    textColor=HexColor("#555555"), spaceAfter=2 * mm, leftIndent=4 * mm,
)
# For markdown headings inside file content
s_md_h1 = ParagraphStyle(
    "MdH1", fontName="CJK", fontSize=12, leading=17,
    textColor=DARK, spaceBefore=4 * mm, spaceAfter=2 * mm,
)
s_md_h2 = ParagraphStyle(
    "MdH2", fontName="CJK", fontSize=11, leading=15,
    textColor=DARK, spaceBefore=3 * mm, spaceAfter=1.5 * mm,
)
s_md_h3 = ParagraphStyle(
    "MdH3", fontName="CJK", fontSize=10, leading=14,
    textColor=HexColor("#333333"), spaceBefore=2.5 * mm, spaceAfter=1 * mm,
)
s_md_h4 = ParagraphStyle(
    "MdH4", fontName="CJK", fontSize=9, leading=13,
    textColor=HexColor("#444444"), spaceBefore=2 * mm, spaceAfter=1 * mm,
)
s_body = ParagraphStyle(
    "Body", fontName="CJK", fontSize=8, leading=12,
    textColor=DARK, spaceAfter=1.5 * mm,
)
s_code = ParagraphStyle(
    "Code", fontName="CJK", fontSize=6.5, leading=9,
    textColor=HexColor("#333333"), backColor=CODE_BG,
    leftIndent=3 * mm, rightIndent=3 * mm,
    spaceBefore=1 * mm, spaceAfter=1 * mm,
    borderColor=HexColor("#cccccc"), borderWidth=0.5, borderPadding=4,
)
s_toc_item = ParagraphStyle(
    "TocItem", fontName="CJK", fontSize=9, leading=13,
    textColor=DARK, leftIndent=6 * mm, spaceAfter=1 * mm,
)
s_separator = ParagraphStyle(
    "Sep", fontName="CJK", fontSize=6, leading=8,
    textColor=HexColor("#cccccc"), alignment=1,
    spaceBefore=4 * mm, spaceAfter=4 * mm,
)

# ---------------------------------------------------------------------------
# Ordered file list  (num, path_or_paths, description)
# path can contain \n for multiple files
# ---------------------------------------------------------------------------
PHASES = [
    ("第一阶段：基础与环境搭建", [
        ("1", "fundamentals/ggml/README.md",
         "GGML 是 llama.cpp 的底层张量库，先理解 tensor、backend、计算图等基本概念"),
        ("2", "fundamentals/llama.cpp/README.md",
         "项目入口：子模块配置、编译方式、GDB 调试、CUDA 构建"),
        ("3", "notes/llama.cpp/debugging.md",
         "调试技巧，后续阅读源码笔记时会频繁用到"),
    ]),
    ("第二阶段：核心概念（推理流水线）", [
        ("4", "fundamentals/llama.cpp/src/tokenize.cpp\nnotes/llama.cpp/llama-vocab-notes.md",
         "分词器 — 推理的第一步"),
        ("5", "fundamentals/llama.cpp/src/simple-prompt.cpp",
         "最简单的推理示例，理解整体调用流程"),
        ("6", "notes/llama.cpp/llama-batch.md",
         "llama_batch 和 llama_ubatch 结构，token 如何组织送入模型"),
        ("7", "notes/llama.cpp/process_ubatch.md",
         "micro-batch 处理细节"),
        ("8", "notes/llama.cpp/kv-cache.md",
         "KV 缓存机制 — 推理加速的核心"),
        ("9", "fundamentals/llama.cpp/src/kv-cache.cpp",
         "KV 缓存的代码实践"),
        ("10", "notes/llama.cpp/graph-inputs.md",
         "计算图输入的构建"),
        ("11", "notes/llama.cpp/gpu-sampling.md",
         "GPU 上的采样实现（temperature、top-k、top-p 等）"),
        ("12", "notes/llama.cpp/output_ids.md",
         "输出 token ID 的处理"),
        ("13", "fundamentals/llama.cpp/src/simple-prompt-multi.cpp",
         "多 prompt 批处理示例"),
    ]),
    ("第三阶段：GPU 加速与后端", [
        ("14", "notes/llama.cpp/cuda.md",
         "CUDA 后端加载机制（ggml_backend_load_all）"),
        ("15", "notes/llama.cpp/cuda-mul-mat.md",
         "CUDA 矩阵乘法实现"),
        ("16", "notes/llama.cpp/cuda-fp16-release-build-issue.md",
         "FP16 构建问题记录"),
        ("17", "fundamentals/ggml/src/llama-att-softmax.cpp",
         "attention softmax 的 GGML 实现"),
        ("18", "notes/llama.cpp/flash-attn-misalignment-issue.md",
         "Flash Attention 对齐问题"),
        ("19", "notes/llama.cpp/macosx.md",
         "macOS (Metal) 平台相关"),
        ("20", "notes/llama.cpp/ggml-threadpool-macos-issue.md",
         "线程池问题"),
    ]),
    ("第四阶段：模型转换与量化", [
        ("21", "notes/llama.cpp/convert.md",
         "convert_hf_to_gguf.py 流程解析"),
        ("22", "notes/llama.cpp/quantization.md",
         "量化原理与 QAT 量化"),
        ("23", "notes/llama.cpp/convert-dequantize.md",
         "反量化过程"),
        ("24", "notes/llama.cpp/devstral2-conversion.md",
         "Devstral2 模型转换实例"),
        ("25", "notes/llama.cpp/convert-mamba-issue.md",
         "Mamba 模型转换问题"),
        ("26", "notes/llama.cpp/gemma-bos-issue.md",
         "Gemma BOS token 问题"),
    ]),
    ("第五阶段：Embeddings", [
        ("27", "fundamentals/llama.cpp/src/embeddings.cpp",
         "embedding 生成代码"),
        ("28", "notes/llama.cpp/embeddings-presets.md",
         "embedding 预设配置"),
    ]),
    ("第六阶段：Server 与部署", [
        ("29", "notes/llama.cpp/llama-server.md",
         "server 启动与 API 调用"),
        ("30", "notes/llama.cpp/server-checkpoints.md",
         "checkpoint 管理"),
        ("31", "notes/llama.cpp/server-logprob-issue.md",
         "log probability 问题"),
        ("32", "notes/llama.cpp/server-unit-tests.md",
         "server 测试"),
        ("33", "notes/llama.cpp/llama-perplexity.md",
         "困惑度计算"),
        ("34", "notes/llama.cpp/tests.md",
         "测试框架"),
        ("35", "notes/llama.cpp/sbatch.md",
         "SLURM 批量提交"),
    ]),
    ("第七阶段：Finetuning", [
        ("36", "fundamentals/llama.cpp/README.md",
         "LoRA 微调、Shakespeare 数据集、chat 格式训练（同文件 #2，此处聚焦 Finetuning 部分）"),
    ]),
    ("第八阶段：多模态与特殊模型", [
        ("37", "notes/llama.cpp/llama-3-2-vision.md",
         "Llama 3.2 视觉模型"),
        ("38", "notes/llama.cpp/qwen-2.5VL-3B-instruct.md",
         "Qwen 视觉模型"),
        ("39", "notes/llama.cpp/vision-model-issue.md",
         "视觉模型问题"),
        ("40", "fundamentals/image-processing/src/mllama.cpp",
         "Llama 视觉模型实现"),
        ("41", "notes/llama.cpp/tts.md",
         "TTS 集成"),
    ]),
    ("第九阶段：Agent 与上层应用", [
        ("42", "agents/llama-cpp-agent/README.md",
         "基于 WASM 的 agent 框架"),
        ("43", "agents/llama-cpp-agent/agent/src/main.rs\nagents/llama-cpp-agent/agent/src/agent.rs\nagents/llama-cpp-agent/agent/src/tool.rs",
         "Rust agent 实现"),
    ]),
    ("第十阶段：其他语言绑定与集成", [
        ("44", "fundamentals/python/src/llama-chat-format.py",
         "Python chat 格式处理"),
        ("45", "fundamentals/rust/llm-chains-llama-example/README.md",
         "Rust LLM chains + Llama"),
        ("46", "fundamentals/rust/llm-chains-chat-demo/src/main-llama.rs",
         "Rust chat demo"),
    ]),
    ("补充：Issue 笔记（按需查阅）", [
        ("A1", "notes/llama.cpp/sched-issue.md",
         "调度问题"),
        ("A2", "notes/llama.cpp/update_chat_msg-issue.md",
         "chat 消息更新问题"),
    ]),
]


def _esc(text: str) -> str:
    """Escape text for XML/reportlab Paragraph."""
    return saxutils.escape(text)


def _is_source_code(path: str) -> bool:
    return path.endswith((".cpp", ".c", ".cc", ".h", ".rs", ".py"))


def _render_markdown(text: str, story: list) -> None:
    """Simple markdown-to-flowables renderer.

    Handles headings, code blocks, and body text.  Not a full parser —
    good enough for the notes in this repo.
    """
    lines = text.split("\n")
    i = 0
    code_buf = []  # type: list
    in_code = False

    while i < len(lines):
        line = lines[i]

        # Code fence
        if line.strip().startswith("```"):
            if in_code:
                # End code block
                code_text = "\n".join(code_buf)
                if code_text.strip():
                    story.append(Preformatted(_esc(code_text), s_code))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Empty line
        if not stripped:
            i += 1
            continue

        # Headings
        if stripped.startswith("####"):
            story.append(Paragraph(_esc(stripped.lstrip("#").strip()), s_md_h4))
            i += 1
            continue
        if stripped.startswith("###"):
            story.append(Paragraph(_esc(stripped.lstrip("#").strip()), s_md_h3))
            i += 1
            continue
        if stripped.startswith("##"):
            story.append(Paragraph(_esc(stripped.lstrip("#").strip()), s_md_h2))
            i += 1
            continue
        if stripped.startswith("#"):
            story.append(Paragraph(_esc(stripped.lstrip("#").strip()), s_md_h1))
            i += 1
            continue

        # Accumulate body paragraph (consecutive non-empty, non-heading lines)
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("#") or nxt.startswith("```"):
                break
            para_lines.append(nxt)
            i += 1

        story.append(Paragraph(_esc(" ".join(para_lines)), s_body))

    # Flush any unclosed code block
    if code_buf:
        story.append(Preformatted(_esc("\n".join(code_buf)), s_code))


def _render_source(text: str, path: str, story: list) -> None:
    """Render a source code file as a preformatted block."""
    ext = os.path.splitext(path)[1].lstrip(".")
    story.append(Paragraph(_esc(f"[{ext}] {os.path.basename(path)}"), s_md_h3))
    story.append(Preformatted(_esc(text), s_code))


def _add_file_content(path: str, story: list) -> None:
    """Read a file and add its content to the story."""
    full = os.path.join(BASE, path)
    if not os.path.isfile(full):
        story.append(Paragraph(f"[文件不存在: {_esc(path)}]", s_body))
        return

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if _is_source_code(path):
        _render_source(text, path, story)
    else:
        _render_markdown(text, story)


# Track files already fully rendered to avoid duplication
_rendered = set()  # type: set


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    story: list = []

    # ---- Cover ----
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("llama.cpp 学习资料", s_title))
    story.append(Paragraph(
        "基于 learning-ai 项目整理 · 从底层到上层的系统学习路径",
        s_subtitle,
    ))
    story.append(Paragraph(
        "整体思路：GGML 基础 → 编译调试 → 推理流水线（tokenize → batch → "
        "KV cache → sampling → output）→ GPU 加速 → 模型转换/量化 → "
        "embedding → server 部署 → finetuning → 多模态 → agent 应用 → "
        "语言绑定。每一层都建立在前一层的理解之上。",
        s_overview,
    ))

    # ---- Table of Contents ----
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("目录", s_phase))
    for phase_title, items in PHASES:
        story.append(Paragraph(f"<b>{_esc(phase_title)}</b>", s_body))
        for num, paths_str, desc in items:
            first_path = paths_str.split("\n")[0]
            story.append(Paragraph(
                f"{num}. {_esc(first_path)}  —  {_esc(desc)}", s_toc_item,
            ))
    story.append(PageBreak())

    # ---- Full content per phase ----
    for phase_title, items in PHASES:
        story.append(Paragraph(phase_title, s_phase))
        story.append(Paragraph(
            "━" * 60, s_separator,
        ))

        for num, paths_str, desc in items:
            paths = [p.strip() for p in paths_str.split("\n") if p.strip()]

            # Section header
            label = f"【{num}】 {paths[0]}"
            if len(paths) > 1:
                label += f"  (+{len(paths)-1} files)"
            story.append(Paragraph(_esc(label), s_file_heading))
            story.append(Paragraph(_esc(desc), s_file_desc))

            for p in paths:
                if p in _rendered:
                    story.append(Paragraph(
                        f"（内容同前，见首次出现处：{_esc(p)}）", s_file_desc,
                    ))
                    continue
                _rendered.add(p)
                _add_file_content(p, story)

            story.append(Spacer(1, 3 * mm))

        story.append(PageBreak())

    doc.build(story)
    print(f"PDF generated: {output_path}")
    sz = os.path.getsize(output_path)
    print(f"Size: {sz:,} bytes ({sz // 1024} KB)")


if __name__ == "__main__":
    build_pdf("llama_cpp_reading_order.pdf")
