import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_sih_presentation(output_path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)

    # Color Palette
    COLOR_PRIMARY = RGBColor(11, 37, 69)      # Deep Navy
    COLOR_SECONDARY = RGBColor(0, 137, 123)   # Teal
    COLOR_ACCENT = RGBColor(230, 81, 0)       # Amber / Orange
    COLOR_DARK = RGBColor(33, 37, 41)         # Charcoal Text
    COLOR_MUTED = RGBColor(108, 117, 125)     # Gray Text
    COLOR_LIGHT_BG = RGBColor(248, 249, 250)  # Light gray background
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_CARD_BORDER = RGBColor(220, 224, 230)

    blank_slide_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_LIGHT_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_badge="SMART INDIA HACKATHON (SIH)"):
        # Top banner line
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = COLOR_ACCENT
        top_bar.line.fill.background()

        # Category badge
        badge = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(5), Inches(0.4))
        tf_b = badge.text_frame
        tf_b.word_wrap = True
        p_b = tf_b.paragraphs[0]
        p_b.text = category_badge.upper()
        p_b.font.size = Pt(11)
        p_b.font.bold = True
        p_b.font.color.rgb = COLOR_SECONDARY

        # Title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

    # ==========================================
    # SLIDE 1: Title & Team Information
    # ==========================================
    slide1 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide1)

    # Accent top bar
    tb1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.2))
    tb1.fill.solid()
    tb1.fill.fore_color.rgb = COLOR_ACCENT
    tb1.line.fill.background()

    # Left card: Title & Problem Details
    left_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.0), Inches(6.8), Inches(5.8))
    left_card.fill.solid()
    left_card.fill.fore_color.rgb = COLOR_WHITE
    left_card.line.color.rgb = COLOR_CARD_BORDER

    tf_lc = left_card.text_frame
    tf_lc.margin_left = Inches(0.4)
    tf_lc.margin_right = Inches(0.4)
    tf_lc.margin_top = Inches(0.4)
    tf_lc.word_wrap = True

    p = tf_lc.paragraphs[0]
    p.text = "SMART INDIA HACKATHON"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    p = tf_lc.add_paragraph()
    p.text = "ConG-MoE v2: Tanglish Sentiment Analysis System"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(14)

    p = tf_lc.add_paragraph()
    p.text = "Contrastive & Discourse-Guided Mixture-of-Experts for Romanized Dravidian Code-Mixed Communications"
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_DARK
    p.space_after = Pt(20)

    p = tf_lc.add_paragraph()
    p.text = "Problem Statement: Real-Time AI Sentiment & Public Grievance Analysis for Regional & Code-Mixed Text"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY
    p.space_after = Pt(6)

    p = tf_lc.add_paragraph()
    p.text = "• Category / Theme: Smart Communication | Artificial Intelligence & NLP\n• Institution: Sri Sivasubramaniya Nadar (SSN) College of Engineering, Chennai"
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_MUTED

    # Right card: Team Members
    right_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(1.0), Inches(4.6), Inches(5.8))
    right_card.fill.solid()
    right_card.fill.fore_color.rgb = COLOR_WHITE
    right_card.line.color.rgb = COLOR_CARD_BORDER

    tf_rc = right_card.text_frame
    tf_rc.margin_left = Inches(0.3)
    tf_rc.margin_right = Inches(0.3)
    tf_rc.margin_top = Inches(0.4)
    tf_rc.word_wrap = True

    p = tf_rc.paragraphs[0]
    p.text = "TEAM COMPOSITION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY
    p.space_after = Pt(10)

    team_members = [
        ("Pragadheeshwar D (Lead)", "AI/NLP Architect & Gated MoE"),
        ("Sriram G S", "Deep Learning & Sequential Models"),
        ("Cibi K", "Full Stack Microservices & APIs"),
        ("Sathiyarajan K", "Corpus Analytics & Evaluation"),
        ("Team Member 5", "Frontend Analytics Dashboard"),
        ("Team Member 6", "DevOps & Streaming Ingestion")
    ]
    for name, role in team_members:
        p = tf_rc.add_paragraph()
        p.text = f"• {name}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY

        p_sub = tf_rc.add_paragraph()
        p_sub.text = f"   Role: {role}"
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = COLOR_MUTED
        p_sub.space_after = Pt(6)


    # ==========================================
    # SLIDE 2: Proposed Solution & Novelty
    # ==========================================
    slide2 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide2)
    add_header(slide2, "Proposed Solution & Key Innovations (USP)")

    # Left: The Challenge / Core Problem
    c1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    c1.fill.solid()
    c1.fill.fore_color.rgb = COLOR_WHITE
    c1.line.color.rgb = COLOR_CARD_BORDER
    tf_c1 = c1.text_frame
    tf_c1.margin_left = tf_c1.margin_right = tf_c1.margin_top = Inches(0.35)
    tf_c1.word_wrap = True

    p = tf_c1.paragraphs[0]
    p.text = "THE PROBLEM: THE ACCURACY PARADOX"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(10)

    bullets1 = [
        ("Majority-Class Skew:", "In vernacular social comments (Tanglish), ~70% are positive. Standard unweighted models achieve 78% accuracy by defaulting ambiguous cases to Positive."),
        ("Minority Sentiment Blindspot:", "Over 75% of mixed/grievance feedback is misclassified as positive, rendering moderation and public grievance tools ineffective."),
        ("Unregulated Orthography:", "Spelling drifts freely (e.g., 'padam / batham') causing 85.8% Out-of-Vocabulary (OOV) rates with standard tokenizers."),
        ("Bipartite Sentiment Flips:", "Sentences like 'Trailer sema aana padam mokka' are mislabeled as positive because bag-of-words ignores adversative conjunctions ('aana').")
    ]
    for title, desc in bullets1:
        p = tf_c1.add_paragraph()
        p.text = f"• {title} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_DARK
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_MUTED
        p.space_after = Pt(8)

    # Right: Our Novel Solution (USP)
    c2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4))
    c2.fill.solid()
    c2.fill.fore_color.rgb = COLOR_WHITE
    c2.line.color.rgb = COLOR_CARD_BORDER
    tf_c2 = c2.text_frame
    tf_c2.margin_left = tf_c2.margin_right = tf_c2.margin_top = Inches(0.35)
    tf_c2.word_wrap = True

    p = tf_c2.paragraphs[0]
    p.text = "OUR SOLUTION: ConG-MoE v2 ARCHITECTURE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY
    p.space_after = Pt(10)

    bullets2 = [
        ("Affect-Preserving Demojization:", "Translates emojis (🔥, 🤮, 💔) into descriptive semantic tokens before stripping, preserving critical sentiment signals."),
        ("Dual-Stream Sub-Word Encodings:", "Combines word n-grams with word-bounded character n-grams (char_wb 2-4), boosting lexical coverage from 14.2% to >99.4%."),
        ("Cost-Sensitive Loss Optimization:", "Imposes a 12.5× higher penalty on minority class misclassifications, directly solving majority bias."),
        ("3-Head Soft Gated Mixture of Experts:", "Pairs a fast base classifier with specialized heads targeting adversative conjunctions ('aana') and emoji polarity conflicts.")
    ]
    for title, desc in bullets2:
        p = tf_c2.add_paragraph()
        p.text = f"✔ {title} "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_MUTED
        p.space_after = Pt(8)


    # ==========================================
    # SLIDE 3: Technical Approach & Architecture
    # ==========================================
    slide3 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide3)
    add_header(slide3, "Technical Approach & System Architecture")

    # Architecture Pipeline Box
    pipe_box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.4))
    pipe_box.fill.solid()
    pipe_box.fill.fore_color.rgb = COLOR_WHITE
    pipe_box.line.color.rgb = COLOR_CARD_BORDER
    tf_pb = pipe_box.text_frame
    tf_pb.margin_left = tf_pb.margin_right = tf_pb.margin_top = Inches(0.35)
    tf_pb.word_wrap = True

    p = tf_pb.paragraphs[0]
    p.text = "END-TO-END SYSTEM PIPELINE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(8)

    steps = [
        ("1. Data Ingestion & Demojization", "Cleans raw social strings, expands colloquial abbreviations, translates emojis into affective tokens (e.g. 🤮 -> EMOJI_DISGUST)."),
        ("2. Dual-Stream Vectorization", "Dual extraction: Word Bigrams for semantic phrases + Character n-grams (char_wb 2-4) to absorb morphological suffixation (-la, -oda)."),
        ("3. Soft Gated Mixture of Experts", "• Head 1 (Base): Cost-sensitive weighted classifier (12.5x penalty)\n• Head 2 (Discourse): Detects contrastive pivot particles ('aana', 'irundhalum')\n• Head 3 (Emoji Conflict): Flags discordance between text & emoji polarity"),
        ("4. Calibrated Output & Delivery", "Produces multi-class probabilities (Positive, Negative, Mixed Feelings, Unknown State) with sub-millisecond latency (<0.33 ms).")
    ]
    for step_title, step_desc in steps:
        p = tf_pb.add_paragraph()
        p.text = step_title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_SECONDARY
        p = tf_pb.add_paragraph()
        p.text = step_desc
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_MUTED
        p.space_after = Pt(6)

    # Right: Tech Stack & Feasibility Specs
    stack_box = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.6), Inches(1.5), Inches(3.9), Inches(5.4))
    stack_box.fill.solid()
    stack_box.fill.fore_color.rgb = COLOR_WHITE
    stack_box.line.color.rgb = COLOR_CARD_BORDER
    tf_sb = stack_box.text_frame
    tf_sb.margin_left = tf_sb.margin_right = tf_sb.margin_top = Inches(0.35)
    tf_sb.word_wrap = True

    p = tf_sb.paragraphs[0]
    p.text = "TECHNOLOGY STACK"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.space_after = Pt(10)

    techs = [
        ("Machine Learning", "Scikit-Learn, PyTorch, NumPy, SciPy"),
        ("Inference Engine", "Serialized ConG-MoE v2 Model (.pkl)"),
        ("Backend Services", "FastAPI (Asynchronous REST API)"),
        ("Dashboard / UI", "Interactive Web App & Terminal CLI"),
        ("Deployment", "Docker Container / CPU-Only Hosting"),
        ("Throughput", ">3,000 comments/sec on single CPU")
    ]
    for category, spec in techs:
        p = tf_sb.add_paragraph()
        p.text = f"{category}:"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_DARK
        p = tf_sb.add_paragraph()
        p.text = spec
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_MUTED
        p.space_after = Pt(6)


    # ==========================================
    # SLIDE 4: Feasibility, Viability & Challenges
    # ==========================================
    slide4 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide4)
    add_header(slide4, "Feasibility, Viability & Key Challenges")

    # Table of Challenges and Mitigations
    rows, cols = 4, 3
    left, top, width, height = Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2)
    table_shape = slide4.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    # Set column widths
    table.columns[0].width = Inches(3.2)
    table.columns[1].width = Inches(3.8)
    table.columns[2].width = Inches(4.7)

    headers = ["Key Operational Challenge", "Practical Risk in Deployment", "Engineered Mitigation Strategy"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE

    challenges_data = [
        ("Severe Class Imbalance\n(~70% Positive comments)",
         "Standard ML defaults to majority class; critical grievances are ignored as 'Positive'.",
         "Cost-Sensitive Learning: Inverse-frequency class weights penalize minority errors 12.5× harder."),
        ("Unregulated Phonetic Spelling\n& Morphological Suffixation",
         "Spelling mutations (padam/batham) yield 85.8% out-of-vocabulary rate with word tokenizers.",
         "Bounded Character n-grams (char_wb 2-4): Learns root morphemes without cross-word noise."),
        ("Bipartite Sentiments &\nContrastive Conjunctions",
         "Bag-of-words ignores shift words like 'aana' (but) and misses polarity flips in comments.",
         "Gated Adversative Expert: Specifically inspects discourse pivot particles to re-weight post-pivot clauses.")
    ]

    for row_idx, (col0, col1, col2) in enumerate(challenges_data, start=1):
        for col_idx, text in enumerate([col0, col1, col2]):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_WHITE if row_idx % 2 == 1 else COLOR_LIGHT_BG
            p = cell.text_frame.paragraphs[0]
            p.text = text
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_DARK


    # ==========================================
    # SLIDE 5: Performance Benchmark & Impact
    # ==========================================
    slide5 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide5)
    add_header(slide5, "Performance Benchmark & Comparative Impact")

    # Table of 5 Models Benchmark
    rows, cols = 6, 7
    left, top, width, height = Inches(0.8), Inches(1.5), Inches(11.7), Inches(3.6)
    table_shape5 = slide5.shapes.add_table(rows, cols, left, top, width, height)
    t5 = table_shape5.table

    t5.columns[0].width = Inches(3.2)
    t5.columns[1].width = Inches(1.3)
    t5.columns[2].width = Inches(1.3)
    t5.columns[3].width = Inches(1.5)
    t5.columns[4].width = Inches(1.5)
    t5.columns[5].width = Inches(1.5)
    t5.columns[6].width = Inches(1.4)

    bench_headers = ["Architecture / Paradigm", "Accuracy", "Macro-F1", "Mixed Recall", "Unknown Recall", "Collapse->Pos", "CPU Latency"]
    for i, h in enumerate(bench_headers):
        cell = t5.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_WHITE

    bench_rows = [
        ("1. TF-IDF + Logistic Reg (Baseline)", "77.60%", "0.5256", "20.55%", "21.66%", "75.00%", "0.007 ms"),
        ("2. BiLSTM (Sequential DL)", "78.86%", "0.6168", "40.24%", "40.07%", "43.84%", "0.555 ms"),
        ("3. CNN + BiLSTM (Shanmugavadivel)", "86.12%", "0.7578", "64.55%", "72.20%", "28.42%", "1.987 ms"),
        ("4. ConG-Hybrid (Our Demojized DL)", "69.30%", "0.6476", "82.19%", "77.62%", "8.73%", "1.870 ms"),
        ("5. ConG-MoE v2 (Our Flagship)", "77.48%", "0.6887", "74.49%", "80.51%", "14.04%", "0.327 ms")
    ]

    for row_idx, row_data in enumerate(bench_rows, start=1):
        is_flagship = (row_idx == 5)
        for col_idx, val in enumerate(row_data):
            cell = t5.cell(row_idx, col_idx)
            cell.fill.solid()
            if is_flagship:
                cell.fill.fore_color.rgb = RGBColor(232, 245, 233) # light mint
            else:
                cell.fill.fore_color.rgb = COLOR_WHITE if row_idx % 2 == 1 else COLOR_LIGHT_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            p.font.bold = is_flagship
            p.font.color.rgb = COLOR_PRIMARY if is_flagship else COLOR_DARK

    # Impact Callout Box below table
    imp_box = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.7))
    imp_box.fill.solid()
    imp_box.fill.fore_color.rgb = COLOR_WHITE
    imp_box.line.color.rgb = COLOR_CARD_BORDER
    tf_ib = imp_box.text_frame
    tf_ib.margin_left = tf_ib.margin_right = tf_ib.margin_top = Inches(0.25)
    tf_ib.word_wrap = True

    p = tf_ib.paragraphs[0]
    p.text = "REAL-WORLD SOCIETAL & OPERATIONAL IMPACT:"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.space_after = Pt(4)

    p = tf_ib.add_paragraph()
    p.text = "• Grievance Identification: Slashes false-positive bias, catching 74.5% of mixed/critical feedback that traditional NLP ignores.\n• Low Cost & Ultra-Fast: Operates at 0.327 ms on commodity CPUs (>3,000 req/sec) without requiring expensive enterprise GPUs.\n• Content Moderation: Accurately flags toxicity, hate speech, and regional sarcasm in digital governance and social media monitoring."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_DARK


    # ==========================================
    # SLIDE 6: Dependencies, Roadmap & Future Scope
    # ==========================================
    slide6 = prs.slides.add_slide(blank_slide_layout)
    set_slide_background(slide6)
    add_header(slide6, "Dependencies, Implementation Roadmap & Scalability")

    # Current Status Box (TRL 5)
    st_box = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.3))
    st_box.fill.solid()
    st_box.fill.fore_color.rgb = COLOR_WHITE
    st_box.line.color.rgb = COLOR_CARD_BORDER
    tf_st = st_box.text_frame
    tf_st.margin_left = tf_st.margin_right = tf_st.margin_top = Inches(0.25)
    tf_st.word_wrap = True

    p = tf_st.paragraphs[0]
    p.text = "CURRENT READINESS STATUS (TRL 5 - VALIDATED IN RELEVANT ENVIRONMENT)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY
    p.space_after = Pt(4)

    p = tf_st.add_paragraph()
    p.text = "✔ Model & Pipeline: Fully trained, benchmarked across 20,000 comments, serialized to lightweight artifacts.\n✔ Working Software Prototype: Interactive Web App (http://localhost:5050) & CLI predictor tested and ready for live demonstration.\n✔ Dependencies: Minimal, open-source stack (Python, Scikit-Learn, FastAPI) with zero proprietary API lock-ins."
    p.font.size = Pt(10)
    p.font.color.rgb = COLOR_MUTED

    # 3 Phase Cards
    phases = [
        ("PHASE 1: HACKATHON MVP", "Current Stage", COLOR_SECONDARY, [
            "Standalone CPU-based ConG-MoE engine",
            "Interactive Web GUI with real-time feedback",
            "Explainable confidence scores & emoji breakdown",
            "Tested on 20,000 annotated comment corpus"
        ]),
        ("PHASE 2: ENTERPRISE INTEGRATION", "Months 1–3", COLOR_PRIMARY, [
            "Streaming connectors (YouTube Data API & Twitter firehose)",
            "Asynchronous message queue (Redis / Celery)",
            "Containerized deployment via Docker / Kubernetes",
            "Automated drift detection & active learning"
        ]),
        ("PHASE 3: MULTILINGUAL SCALE", "Months 4–6", COLOR_ACCENT, [
            "Zero-shot transfer to Manglish & Kanglish",
            "Civic grievance dashboard for public administration",
            "Speech-to-Text integration via Whisper-Indic",
            "Edge deployment on low-cost IoT / Raspberry Pi"
        ])
    ]

    for idx, (p_title, p_time, p_color, items) in enumerate(phases):
        card = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8 + idx * 4.0), Inches(3.0), Inches(3.7), Inches(3.9))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_WHITE
        card.line.color.rgb = COLOR_CARD_BORDER

        tf_c = card.text_frame
        tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = Inches(0.25)
        tf_c.word_wrap = True

        p = tf_c.paragraphs[0]
        p.text = p_title
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = p_color

        p_sub = tf_c.add_paragraph()
        p_sub.text = p_time
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = COLOR_MUTED
        p_sub.space_after = Pt(8)

        for item in items:
            p_item = tf_c.add_paragraph()
            p_item.text = f"• {item}"
            p_item.font.size = Pt(9.5)
            p_item.font.color.rgb = COLOR_DARK
            p_item.space_after = Pt(4)

    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "SIH_Presentation_ConG_MoE.pptx"
    create_sih_presentation(out)
