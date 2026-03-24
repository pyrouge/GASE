"""
Script to generate 3 sample PDF documents with different structures.
Run: python create_sample_docs.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime


def create_financial_report():
    """Creates a sample financial report document."""
    output_file = "data/sample_documents/financial_report_2023.pdf"
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=1
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("ANNUAL FINANCIAL REPORT 2023", title_style))
    elements.append(Paragraph("Acme Corporation Inc.", styles['Heading3']))
    elements.append(Spacer(1, 12))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading2']))
    elements.append(Paragraph(
        "Our financial performance in 2023 was strong, with revenues increasing by 23% year-over-year. "
        "Net income reached $450M, up from $365M in 2022. We maintained a solid balance sheet with "
        "cash reserves of $1.2B, allowing us to invest in R&D and return capital to shareholders.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Financial Highlights section
    elements.append(Paragraph("Financial Highlights", styles['Heading2']))
    data = [
        ['Metric', '2023', '2022', 'YoY Change'],
        ['Revenue (M)', '$1,950', '$1,585', '+23.0%'],
        ['Net Income (M)', '$450', '$365', '+23.3%'],
        ['Operating Margin', '23.1%', '23.0%', '+0.1pp'],
        ['EPS', '$2.25', '$1.82', '+23.6%'],
    ]
    t = Table(data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # Revenue Analysis
    elements.append(Paragraph("Revenue Analysis", styles['Heading2']))
    elements.append(Paragraph(
        "Total revenue for 2023 was $1,950M compared to $1,585M in 2022, representing a 23% increase. "
        "This growth was driven by strong demand in Q3 2023, where revenue reached $520M, "
        "compared to Q3 2022's $420M. Q4 2023 also performed well with $490M in revenue.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Quarterly Revenue Breakdown", styles['Heading3']))
    elements.append(Paragraph(
        "Q1 2023: $450M (vs $385M in 2021) | "
        "Q2 2023: $490M (vs $390M in 2021) | "
        "Q3 2023: $520M (vs $420M in 2021) | "
        "Q4 2023: $490M (vs $390M in 2021)",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Profitability
    elements.append(Paragraph("Profitability", styles['Heading2']))
    elements.append(Paragraph(
        "Net income in 2023 reached $450M, an increase of 23.3% from 2022. "
        "Our operating margin remained stable at 23.1%, demonstrating operational efficiency. "
        "The company's gross margin improved to 45% from 44% in 2022 due to favorable input costs.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Balance Sheet
    elements.append(Paragraph("Balance Sheet Summary", styles['Heading2']))
    elements.append(Paragraph(
        "Total assets as of December 31, 2023 were $8.5B, an increase from $7.8B in 2022. "
        "Current assets totaled $3.2B with cash and equivalents at $1.2B. "
        "Total liabilities were $4.1B, including $2.5B in debt at an average interest rate of 3.5%.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Risk Factors
    elements.append(PageBreak())
    elements.append(Paragraph("Risk Factors", styles['Heading2']))
    elements.append(Paragraph(
        "The company faces several risks in 2024. Market competition is increasing, with new entrants "
        "reducing our market share. Supply chain disruptions, though improving, remain a concern. "
        "Interest rate increases could impact our debt servicing costs. Currency fluctuations "
        "in international markets could affect our reported earnings by 2-3%.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Outlook
    elements.append(Paragraph("2024 Outlook", styles['Heading2']))
    elements.append(Paragraph(
        "For fiscal 2024, we project revenue growth of 15-18%, with net income expected to grow "
        "at a similar rate. We will continue investing in R&D at 12% of revenue and maintain "
        "our dividend payout ratio at 30-35% of net income.",
        styles['BodyText']
    ))
    
    doc.build(elements)
    print(f"✓ Created: {output_file}")


def create_research_paper():
    """Creates a sample research paper document."""
    output_file = "data/sample_documents/research_paper_ai.pdf"
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1
    )
    
    elements = []
    
    # Title and authors
    elements.append(Paragraph("Graph-Augmented Retrieval in Large Language Models", title_style))
    elements.append(Paragraph("Dr. Jane Smith<sup>1</sup>, Prof. John Doe<sup>2</sup>, Dr. Alice Johnson<sup>1</sup>", 
                            styles['Normal']))
    elements.append(Paragraph("<i><sup>1</sup>Institute for AI Research, <sup>2</sup>Tech University</i>", 
                            styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Abstract
    elements.append(Paragraph("Abstract", styles['Heading2']))
    elements.append(Paragraph(
        "This paper presents GASE, a novel Graph-Augmented Structural Ensemble approach for improving "
        "retrieval-augmented generation in large language models. Our method combines semantic, lexical, "
        "and structural signals to achieve 23% improvement in context precision over baseline methods. "
        "We evaluate on FinanceBench and CUAD datasets, demonstrating superiority in structure-dependent queries.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Introduction
    elements.append(Paragraph("1. Introduction", styles['Heading2']))
    elements.append(Paragraph(
        "Retrieval-augmented generation (RAG) has emerged as a key technique for improving LLM performance "
        "on knowledge-intensive tasks. However, traditional RAG systems treat documents as bags of sentences, "
        "ignoring structural relationships between passages. This paper addresses this limitation by proposing "
        "a structure-aware retrieval system that explicitly models document hierarchy.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Related Work
    elements.append(Paragraph("2. Related Work", styles['Heading2']))
    elements.append(Paragraph(
        "Prior work on semantic search includes Dense Passage Retrieval (DPR) and ColBERT, which focus on "
        "dense embeddings. BM25 remains the standard for lexical matching. GraphRAG (Microsoft) and LightRAG "
        "introduce entity-based graph structures, but do not prioritize document hierarchy. HiRAG proposes "
        "hierarchical retrieval but lacks comprehensive evaluation on real-world documents.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Methodology
    elements.append(Paragraph("3. Methodology", styles['Heading2']))
    elements.append(Paragraph("3.1 Document Parsing", styles['Heading3']))
    elements.append(Paragraph(
        "We use Docling to extract document structure from PDFs. Each header creates a node in our hierarchy, "
        "with breadcrumb paths encoding the full structural context (e.g., 'Risks > Market > Volatility').",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 6))
    
    elements.append(Paragraph("3.2 Multi-Signal Retrieval", styles['Heading3']))
    elements.append(Paragraph(
        "Our retrieval pipeline runs three independent signals: (1) BM25 for keyword matching, "
        "(2) dense embeddings via SBERT for semantic similarity, (3) graph traversal for structural expansion.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 6))
    
    elements.append(Paragraph("3.3 Fusion Algorithm", styles['Heading3']))
    elements.append(Paragraph(
        "Results are combined via weighted linear combination: R(n) = α·vector + β·bm25 + γ·authority. "
        "Authority scores encode section importance and neighbor density in the document graph.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Evaluation
    elements.append(PageBreak())
    elements.append(Paragraph("4. Experimental Results", styles['Heading2']))
    elements.append(Paragraph(
        "We evaluated GASE on 50 structure-dependent queries from FinanceBench and CUAD. "
        "Our method achieved 78% context precision, compared to 55% for naive vector-only retrieval. "
        "Latency remained sub-second for all queries.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Conclusion
    elements.append(Paragraph("5. Conclusion and Future Work", styles['Heading2']))
    elements.append(Paragraph(
        "We presented GASE, demonstrating that structural hierarchy is a valuable signal for document retrieval. "
        "Future work includes entity graph overlay, multi-document QA, and auto-tuning of fusion weights.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # References
    elements.append(Paragraph("References", styles['Heading2']))
    references = [
        "[1] Karpukhin et al., Dense Passage Retrieval for Open-Domain Question Answering, 2020",
        "[2] Li et al., GraphRAG: Local and Global Graph Retrieval-Augmented Generation, 2024",
        "[3] Khattab & Zaharia, ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction, 2020",
    ]
    for ref in references:
        elements.append(Paragraph(ref, styles['BodyText']))
    
    doc.build(elements)
    print(f"✓ Created: {output_file}")


def create_legal_document():
    """Creates a sample legal document."""
    output_file = "data/sample_documents/contract_terms.pdf"
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("SERVICE AGREEMENT", title_style))
    elements.append(Paragraph("Effective Date: January 1, 2024", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Parties
    elements.append(Paragraph("1. PARTIES", styles['Heading2']))
    elements.append(Paragraph(
        "This Agreement is entered into between TechCorp Inc., a Delaware corporation (\"Provider\"), "
        "and ClientCo LLC, an Ohio limited liability company (\"Client\"). "
        "The Provider and Client are collectively referred to as the \"Parties.\"",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Scope
    elements.append(Paragraph("2. SCOPE OF SERVICES", styles['Heading2']))
    elements.append(Paragraph(
        "Provider agrees to provide software development and consulting services as defined in the "
        "Statement of Work (SOW) attached as Exhibit A. Services include system design, development, "
        "testing, and deployment. Client shall designate a Project Manager responsible for all communications.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Payment Terms
    elements.append(Paragraph("3. PAYMENT AND FEES", styles['Heading2']))
    elements.append(Paragraph("3.1 Base Fees", styles['Heading3']))
    elements.append(Paragraph(
        "Client shall pay Provider $150,000 per month for the base service package. "
        "Additional development hours beyond 100 hours per month shall be billed at $200 per hour. "
        "Payment is due within 30 days of invoice.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 6))
    
    elements.append(Paragraph("3.2 Expenses", styles['Heading3']))
    elements.append(Paragraph(
        "Reasonable out-of-pocket expenses including travel, software licenses, and cloud infrastructure "
        "shall be reimbursed at cost plus 10% administrative fee. Travel expenses require prior approval.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Term and Termination
    elements.append(Paragraph("4. TERM AND TERMINATION", styles['Heading2']))
    elements.append(Paragraph(
        "This Agreement shall commence on January 1, 2024 and continue for a period of two (2) years "
        "unless earlier terminated. Either party may terminate for convenience with 30 days written notice. "
        "Termination for cause (material breach) requires 10 days cure period.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Confidentiality
    elements.append(PageBreak())
    elements.append(Paragraph("5. CONFIDENTIALITY", styles['Heading2']))
    elements.append(Paragraph(
        "Each party agrees to maintain the confidentiality of proprietary information disclosed by the other party. "
        "Confidential information shall not be disclosed to third parties without written consent. "
        "Obligations under this section survive termination for a period of three (3) years.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Intellectual Property
    elements.append(Paragraph("6. INTELLECTUAL PROPERTY", styles['Heading2']))
    elements.append(Paragraph(
        "Work product created by Provider using Client's specifications and materials shall be owned by Client. "
        "Pre-existing materials and tools created independently by Provider remain Provider's property. "
        "Client retains a royalty-free license to use Provider's tools.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Warranties and Disclaimers
    elements.append(Paragraph("7. WARRANTIES AND DISCLAIMERS", styles['Heading2']))
    elements.append(Paragraph(
        "Provider warrants that all services shall be performed in a professional and workmanlike manner "
        "in accordance with industry standards. However, Provider disclaims all other warranties, "
        "express or implied, including merchantability and fitness for a particular purpose.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Limitation of Liability
    elements.append(Paragraph("8. LIMITATION OF LIABILITY", styles['Heading2']))
    elements.append(Paragraph(
        "In no event shall either party's total liability exceed the total fees paid or payable in the "
        "12 months preceding the claim. Neither party shall be liable for indirect, incidental, special, "
        "or consequential damages, including lost profits or data loss.",
        styles['BodyText']
    ))
    elements.append(Spacer(1, 12))
    
    # Governing Law
    elements.append(Paragraph("9. GOVERNING LAW AND DISPUTE RESOLUTION", styles['Heading2']))
    elements.append(Paragraph(
        "This Agreement shall be governed by the laws of the State of California without regard to conflict principles. "
        "Any disputes shall first be addressed through good faith negotiation. If unresolved within 30 days, "
        "either party may pursue binding arbitration under JAMS rules.",
        styles['BodyText']
    ))
    
    doc.build(elements)
    print(f"✓ Created: {output_file}")


if __name__ == "__main__":
    import os
    os.makedirs("data/sample_documents", exist_ok=True)
    
    try:
        create_financial_report()
        create_research_paper()
        create_legal_document()
        print("\n✓ All sample PDFs created successfully!")
    except Exception as e:
        print(f"✗ Error creating PDFs: {e}")
        print("\nRequired: pip install reportlab")
